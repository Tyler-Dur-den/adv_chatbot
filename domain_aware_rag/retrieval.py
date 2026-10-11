import os
import re
import logging
import chromadb

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

from domain_aware_rag.config import RAGConfig, settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

def _sanitize_collection_name(raw_name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
    if len(cleaned) < 3:
        cleaned = cleaned.ljust(3, '_')
    return cleaned[:63]

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)

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