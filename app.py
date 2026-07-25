import os
import tempfile
import streamlit as st
from main import get_retriever_for_doc_type, create_rag_chain

st.set_page_config(page_title="Domain-Aware RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Domain-Aware Hybrid RAG System")
st.caption("Upload a PDF, select its domain, and ask questions with domain-optimized retrieval.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.header("⚙️ Configuration")
    
    doc_type = st.selectbox(
        "Select Document Type",
        ("Financial Report", "Legal Contract", "Academic Paper", "Technical Manual"),
        help="Selects custom chunking and retrieval strategies designed for this document structure."
    )
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing & Indexing Document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            retriever = get_retriever_for_doc_type(tmp_path, doc_type)
            st.session_state.rag_chain = create_rag_chain(retriever)

            os.remove(tmp_path)
            st.success(f"Document indexed as **{doc_type}**!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about your document..."):
    if not st.session_state.rag_chain:
        st.warning("Please upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching context & generating answer..."):
                response = st.session_state.rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                
                with st.expander("🔍 View Source Documents & Citations"):
                    for idx, doc in enumerate(response["context"]):
                        st.markdown(f"**Chunk {idx+1} (Page {doc.metadata.get('page', 'N/A')})**")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})