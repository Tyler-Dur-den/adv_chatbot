from retrieval import get_retriever_for_doc_type
from graph import build_rag_graph

def main():
    print("Loading document and initializing Advanced RAG pipeline...")
    # Initialize the retriever
    retriever = get_retriever_for_doc_type("2312.10997v5.pdf", "Academic Paper")
    
    # Build and compile the LangGraph application
    app = build_rag_graph(retriever)
    
    # Required for LangGraph's MemorySaver checkpointer
    config = {"configurable": {"thread_id": "1"}}
    
    print("\nPipeline Ready. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        state = {"messages": [("user", user_input)]}
        
        try:
            result = app.invoke(state, config=config)
            print(f"\nAI: {result['messages'][-1].content}")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()