from domain_aware_rag.retrieval import get_retriever_for_doc_type

def hit_at_k(docs, gold_pages, k=3):
    for rank, d in enumerate(docs[:k], 1):
        if d.metadata.get("page") in gold_pages:
            return rank
    return None

golden_dataset = [
    {
        "query": "What is the primary difference between Naive RAG and Advanced RAG?",
        "gold_pages": [1, 2, 3] 
    },
    {
        "query": "How does Query Transformation improve chunk retrieval?",
        "gold_pages": [8, 9, 10] 
    },
    {
        "query": "What are the core metrics used to evaluate the generation aspect of a RAG system?",
        "gold_pages": [13, 14, 15] 
    },
    {
        "query": "Why is optimizing retrieval granularity important during inference?",
        "gold_pages": [7, 8]
    },
    {
        "query": "What defines a Modular RAG architecture compared to previous iterations?",
        "gold_pages": [3, 4]
    }
]

def run_evals():
    print("Ingesting PDF and building vector index (this takes a minute)...")
    retriever = get_retriever_for_doc_type("2312.10997v5.pdf", "Academic Paper")
    
    hits = 0
    
    for idx, item in enumerate(golden_dataset):
        print(f"\n--- Test {idx+1} ---")
        print(f"Query: {item['query']}")
        
        # Fetch the re-ranked chunks
        docs = retriever.invoke(item["query"])
        
        hit_rank = hit_at_k(docs, item["gold_pages"], k=3)
        
        if hit_rank:
            print(f"✅ HIT at Rank {hit_rank}")
            hits += 1
        else:
            top_page = docs[0].metadata.get('page') if docs else "N/A"
            print(f"❌ MISS - Top chunk was from page {top_page}")
            
    print(f"\nFinal Retrieval Accuracy: {hits}/{len(golden_dataset)} ({(hits/len(golden_dataset))*100}%)")

if __name__ == "__main__":
    run_evals()