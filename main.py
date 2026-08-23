import os
import re
import logging
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

class RAGConfig:
    MAX_CONTEXT_CHARS = 12000
    PERSIST_DIR = "./chroma_data"
    
    DOC_CHUNK_CONFIG = {
        "Financial Report": {"chunk_size": 1000, "overlap": 100, "k": 4},
        "Legal Contract": {"chunk_size": 800, "overlap": 200, "k": 3},
        "Academic Paper": {"use_semantic": True, "k": 3, "fetch_k": 20},
        "default": {"chunk_size": 500, "overlap": 50, "k": 3}
    }

def _extract_text(content) -> str:
    if content is None:
        return ""
    if hasattr(content, "content"):
        content = content.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "content"):
                parts.append(str(item.content))
            else:
                parts.append(str(item))
        return " ".join([p for p in parts if p])
    if isinstance(content, dict):
        return str(content.get("text", str(content)))
    return str(content)

def _sanitize_collection_name(raw_name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
    if len(cleaned) < 3:
        cleaned = cleaned.ljust(3, '_')
    return cleaned[:63]

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=RAGConfig.PERSIST_DIR)

def get_retriever_for_doc_type(file_path: str, doc_type: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    config = RAGConfig.DOC_CHUNK_CONFIG.get(doc_type, RAGConfig.DOC_CHUNK_CONFIG["default"])
    client = get_chroma_client()
    
    raw_coll_name = f"{doc_type}_{os.path.basename(file_path)}"
    collection_name = _sanitize_collection_name(raw_coll_name)

    if doc_type == "Academic Paper":
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        split_docs = splitter.split_documents(docs)
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            client=client,
            collection_name=collection_name
        )
        base_retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": config["k"], "fetch_k": config.get("fetch_k", 10)}
        )
        
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["overlap"]
        )
        split_docs = splitter.split_documents(docs)
        
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            client=client,
            collection_name=collection_name
        )
        
        if doc_type == "Legal Contract":
            similarity_retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": config["k"]}
            )
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = config["k"]
            
            base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, similarity_retriever], 
                weights=[0.5, 0.5]
            )
        else:
            base_retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": config["k"]}
            )

    reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=config["k"])
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    standalone_question: Optional[str]
    context: List[Document]

def build_rag_graph(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    def contextualize_question_node(state: RAGState) -> dict:
        messages = state["messages"]
        latest_question = _extract_text(messages[-1])
        if len(messages) <= 1:
            return {"standalone_question": latest_question}

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Given chat history and a follow-up question, rewrite it "
                "into a standalone question that contains all necessary context. "
                "Return ONLY the reformulated question, nothing else."
            )),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}")
        ])
        chain = prompt | llm
        try:
            reformulated = chain.invoke({
                "chat_history": messages[:-1],
                "input": latest_question
            })
            return {"standalone_question": _extract_text(reformulated)}
        except Exception as e:
            logger.error(f"Contextualization failed: {e}", exc_info=True)
            return {"standalone_question": latest_question}

    def retrieve_node(state: RAGState) -> dict:
        raw_query = state.get("standalone_question")
        if not raw_query:
            raw_query = state["messages"][-1]
            
        query = str(_extract_text(raw_query)).strip()
        try:
            docs = retriever.invoke(query)
            if not docs:
                return {
                    "context": [Document(
                        page_content="The document does not contain information relevant to your question.",
                        metadata={}
                    )]
                }
            return {"context": docs}
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            return {
                "context": [Document(
                    page_content="I encountered an error accessing the document store.",
                    metadata={"error": str(e)}
                )]
            }
    def generate_answer_node(state: RAGState) -> dict:
        docs = state["context"]
        original_question = _extract_text(state["messages"][-1])
        formatted_context = "\n\n---\n\n".join(
            f"[Source {i+1}]: {doc.page_content}" 
            for i, doc in enumerate(docs[:6])
        )
        if len(formatted_context) > RAGConfig.MAX_CONTEXT_CHARS:
            formatted_context = formatted_context[:RAGConfig.MAX_CONTEXT_CHARS] + "\n[Context truncated due to length]"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert document analysis assistant. "
                "Answer the user's question directly and authoritatively using the provided context.\n"
                "- Cite sources explicitly: [Source X]\n"
                "- If the context lacks sufficient information, say so clearly\n"
                "- Do NOT hallucinate details not present in the documents\n\n"
                "CONTEXT:\n{context}"
            )),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        try:
            response = chain.invoke({
                "context": formatted_context,
                "chat_history": state["messages"][:-1],
                "input": original_question
            })
            return {"messages": [AIMessage(content=_extract_text(response))]}
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return {
                "messages": [AIMessage(
                    content=f"Generation Error: {str(e)}"
                )]
            }

    builder = StateGraph(RAGState)
    builder.add_node("contextualize", contextualize_question_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_answer_node)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)