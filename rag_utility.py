import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

def process_document_to_faiss(uploaded_files):
    """Processes multiple PDFs and merges them into one FAISS index."""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(working_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loader = UnstructuredPDFLoader(temp_path)
            documents = loader.load()
            # Ensure metadata tracks the filename
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
            
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if all_chunks:
        vectordb = FAISS.from_documents(all_chunks, embedding)
        vectordb.save_local(os.path.join(working_dir, "faiss_index"))
        return True
    return False

def answer_question(user_question):
    """Modern LCEL RAG chain that returns answer and source documents."""
    try:
        vectordb = FAISS.load_local(
            os.path.join(working_dir, "faiss_index"), 
            embedding,
            allow_dangerous_deserialization=True
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate.from_template("""
        Answer based only on the context provided:
        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """)

        # LCEL chain returning a dictionary with context and answer
        rag_chain = (
            RunnableParallel({
                "context": retriever | format_docs, 
                "source_docs": retriever,
                "question": RunnablePassthrough()
            })
            | {
                "answer": prompt | llm | StrOutputParser(),
                "sources": lambda x: list(set(doc.metadata.get("source", "Unknown") for doc in x["source_docs"]))
            }
        )
        return rag_chain.invoke(user_question)
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}