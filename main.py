from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

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

    doc = splitter.split_documents(financial_doc)

    vector_store = Chroma.from_documents(doc,embeddings)
    return vector_store.as_retriever(search_type ="similarity", search_kwargs={"k":4})

def legal_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    doc = splitter.split_documents(legal_doc)

    vector_store = Chroma.from_documents(doc,embeddings)
    similarity_retreiver = vector_store.as_retriever(search_type="similarity",search={"k":3})

    bm25_retriever = BM25Retriever.from_documents(doc)
    bm25_retriever.k = 3

    return EnsembleRetriever(retrievers=[similarity_retreiver,bm25_retriever], weights=[0.5,0.5])
