# 📚 Domain-Aware Hybrid RAG System

An adaptive document intelligence system that selects optimised 
retrieval strategies based on document type, powered by LangGraph 
for stateful multi-turn conversations.

## The Problem

Generic RAG systems use one-size-fits-all chunking and retrieval — 
a legal contract needs exact keyword matching, an academic paper 
needs semantic understanding, a financial report needs precise 
numerical context. Applying the same strategy to all document 
types degrades retrieval quality significantly.

## How It Works

Upload a document, select its type, and ask questions in natural 
language. The system:

1. Selects a domain-optimised chunking and retrieval strategy 
   based on document type
2. Contextualises follow-up questions using chat history — 
   "what did it say about that?" becomes a standalone searchable query
3. Retrieves relevant chunks using hybrid or semantic search
4. Re-ranks results using a cross-encoder for precision
5. Generates a cited answer grounded in document context

## Domain-Specific Strategies

| Document Type | Chunking | Retrieval | Why |
|---|---|---|---|
| Financial Report | Fixed 1000 chars | Semantic similarity | Numerical context needs broad chunks |
| Legal Contract | Fixed 800 chars + overlap | BM25 + Semantic hybrid | Exact clause matching + semantic understanding |
| Academic Paper | Semantic chunking | MMR | Avoids redundant chunks, finds diverse evidence |
| Technical Manual | Fixed 500 chars | Semantic similarity | Short precise chunks for specific answers |

## Architecture

```
User Question
      ↓
Contextualize Node (LangGraph)
Rewrites follow-up questions into standalone queries using chat history
      ↓
Retrieve Node
Domain-aware retriever → Cross-encoder reranker (BAAI/bge-reranker-v2-m3)
      ↓
Generate Node
Gemini Flash generates cited answer with [Source X] references
      ↓
MemorySaver Checkpoint
Full conversation state persisted across turns per thread ID
```

## Tech Stack

- **LangGraph** — stateful multi-node RAG pipeline with MemorySaver 
  checkpointing for persistent conversation memory
- **Google Gemini Flash** — LLM for question contextualization and 
  answer generation
- **HuggingFace Embeddings** — all-MiniLM-L6-v2 for vector embeddings
- **ChromaDB** — persistent vector store with named collections per document
- **BM25Retriever** — keyword-based sparse retrieval for legal documents
- **EnsembleRetriever** — hybrid sparse-dense retrieval
- **BAAI/bge-reranker-v2-m3** — cross-encoder reranking to filter 
  noisy chunks before generation
- **SemanticChunker** — boundary-aware chunking for academic papers
- **Streamlit** — interactive web interface

## Key Features

- **Question contextualization** — follow-up questions are rewritten 
  into standalone queries before retrieval, maintaining coherent 
  multi-turn conversations
- **Source citations** — every answer cites [Source X] mapped to 
  exact document chunks
- **Persistent memory** — MemorySaver checkpointer maintains full 
  conversation state per session thread ID
- **Cross-encoder reranking** — retrieved chunks are re-scored by 
  a cross-encoder before generation, filtering irrelevant results

## Setup

1. Clone the repo
```bash
git clone https://github.com/Tyler-Dur-den/adv_chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file
```
GOOGLE_API_KEY=your_gemini_key
```

4. Run the app
```bash
streamlit run app.py
```

## Live Demo
[Try it here](https://advchatbot-9fhikxbwsbyhqrjr2znbgm.streamlit.app)

## Limitations
- Scanned PDFs not supported — digital text PDFs only
- Large documents may hit Gemini Flash context limits
- Cross-encoder reranking adds latency on first query per session