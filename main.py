from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

def get_retriever_for_doc_type(file_path: str, doc_type: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if doc_type == "Financial Report":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        vector_store = Chroma.from_documents(split_docs, embeddings)
        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=4)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
        
    elif doc_type == "Legal Contract":
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        vector_store = Chroma.from_documents(split_docs, embeddings)
        similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        bm25_retriever = BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, similarity_retriever], weights=[0.5, 0.5])
        reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=4)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)
    
    elif doc_type == "Academic Paper":
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        split_docs = splitter.split_documents(docs)
        vector_store = Chroma.from_documents(split_docs, embeddings)
        base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20})
        reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=4)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
    
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        vector_store = Chroma.from_documents(split_docs, embeddings)
        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=4)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
    
def create_rag_chain(retriever):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    system_prompt = (
        "You are an expert document assistant. Use the following pieces of retrieved "
        "context to answer the question. If you don't know the answer, say that you don't know. "
        "Keep the answer detailed, professional, and clear.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)