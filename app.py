import os
import tempfile
import uuid
import streamlit as st
from main import get_retriever_for_doc_type, build_rag_graph
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Domain-Aware RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Domain-Aware Hybrid RAG System")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("⚙️ Configuration")
    doc_type = st.selectbox(
        "Select Document Type",
        ("Financial Report", "Legal Contract", "Academic Paper", "Technical Manual")
    )
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Indexing Document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            retriever = get_retriever_for_doc_type(tmp_path, doc_type)
            st.session_state.chatbot = build_rag_graph(retriever)
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            
            os.remove(tmp_path)
            st.success("Document Indexed Successfully!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

if user_input := st.chat_input("Ask a question about your document..."):
    if not st.session_state.chatbot:
        st.warning("Please upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing through graph..."):
                output = st.session_state.chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG
                )
                answer = output["messages"][-1].content
                st.markdown(answer)

                with st.expander("🔍 View Retrieved Documents"):
                    for idx, doc in enumerate(output.get("context", [])):
                        st.markdown(f"**Chunk {idx+1} (Page {doc.metadata.get('page', 'N/A')})**")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})  