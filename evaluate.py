from retrieval import get_retriever_for_doc_type

golden_dataset = [
    {
        "query": "What is the primary difference between Naive RAG and Advanced RAG?",
        "expected_keyword": "pre-retrieval"
    },
    {
        "query": "How does Query Transformation improve chunk retrieval?",
        "expected_keyword": "transformed"
    },
    {
        "query": "What are the core metrics used to evaluate the generation aspect of a RAG system?",
        "expected_keyword": "Context Relevance"
    },
    {
        "query": "Why is optimizing retrieval granularity important during inference?",
        "expected_keyword": "granularity"
    },
    {
        "query": "What defines a Modular RAG architecture compared to previous iterations?",
        "expected_keyword": "interchangeable"
    }
]

def run_evals():
    print("Ingesting PDF and building vector index (this takes a minute)...")
    retriever = get_retriever_for_doc_type("2312.10997v5.pdf", "Academic Paper")
    
    hits = 0
    
    for i, item in enumerate(golden_dataset):
        print(f"\n--- Test {i+1} ---")
        print(f"Query: {item['query']}")
        
        docs = retriever.invoke(item["query"])
        
        hit = False
        for rank, doc in enumerate(docs):
            if item["expected_keyword"].lower() in doc.page_content.lower():
                hit = True
                print(f"✅ HIT at Rank {rank+1} (Length: {len(doc.page_content)})")
                break
        
        if not hit:
            print("❌ MISS - Top chunk retrieved was:")
            print(docs[0].page_content[:200]) 
        else:
            hits += 1
            
    print(f"\nFinal Retrieval Accuracy: {hits}/{len(golden_dataset)} ({(hits/len(golden_dataset))*100}%)")

if __name__ == "__main__":
    run_evals()