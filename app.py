import os
import streamlit as st
from rag_utility import process_document_to_faiss, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("🤖 Multi-PDF Knowledge Assistant")

# Use a specific key to prevent reload issues
uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Indexing documents..."):
            success = process_document_to_faiss(uploaded_files)
            if success:
                st.success("Documents Processed Successfully!")
    else:
        st.warning("Please upload files first.")

user_question = st.text_input("Ask your question about the documents")

if st.button("Answer"):
    if user_question:
        with st.spinner("Generating answer..."):
            result = answer_question(user_question)
            
            st.markdown("### Llama-3.3-70B Response")
            st.write(result["answer"])
            
            if result.get("sources"):
                st.markdown("### Sources")
                for src in result["sources"]:
                    st.info(f"📄 {src}")
    else:
        st.warning("Please enter a question.")