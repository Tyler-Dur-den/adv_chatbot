from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
load_dotenv()
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)   

financial_path = "tsla-20231231.pdf"
loader = PyPDFLoader(financial_path)
financial_doc = loader.load()

legal_path = "ex103to8ka07380004_11132013.pdf"
loader = PyPDFLoader(legal_path)
legal_doc = loader.load()

academic_path = "1706.03762v7.pdf"
loader = PyPDFLoader(academic_path)
academic_doc = loader.load()

technical_path = "RP-008344-DS-5-raspberry-pi-4-product-brief.pdf"
loader = PyPDFLoader(technical_path)
technical_doc = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def financial_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        separators=["\n\n", "\n", " ", ""]
    )

    doc = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(doc,embeddings)
    return vector_store.as_retriever(search_type ="similarity", search_kwargs={"k":4})

def legal_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    doc = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(doc,embeddings)
    similarity_retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})

    bm25_retriever = BM25Retriever.from_documents(doc)
    bm25_retriever.k = 3

    base_hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, similarity_retriever], 
        weights=[0.5, 0.5]
    )

    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=base_hybrid_retriever
    )

    return compressed_retriever

def academic_splitter(docs):
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    doc = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(doc,embeddings)
    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20})

    compressed_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
    
def techinal_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
        )
    doc = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(doc,embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def create_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    system_prompt = (
        "You are an document assistant. Use the following pieces of retrieved "
        "context to answer the question. If you don't know the answer, say that you don't know. "
        "Keep the answer detailed, professional, and clear.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

