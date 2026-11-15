import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter


ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
load_dotenv(ROOT_DIR / ".env")

st.set_page_config(page_title="Simple RAG Chat", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner="Building the document index...")
def build_retriever(document_path: str):
    loader = TextLoader(document_path, encoding="utf-8")
    loaded_document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks_of_text = text_splitter.split_documents(loaded_document)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.from_documents(chunks_of_text, embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3})


@st.cache_resource
def build_chain(document_path: str, model_name: str):
    retriever = build_retriever(document_path)
    template = """Answer the question based only on the following context:

{context}

Question: {question}

If the answer is not in the context, say that the document does not provide enough information.
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatGroq(model=model_name, temperature=0)

    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


st.title("Simple RAG Chat")
st.caption("Ask questions about the text documents in the data folder.")

document_paths = sorted(DATA_DIR.glob("*.txt"))
if not document_paths:
    st.error("No .txt documents were found in the data folder.")
    st.stop()

selected_path = st.selectbox(
    "Document",
    document_paths,
    format_func=lambda path: path.name,
)

if not os.getenv("GROQ_API_KEY"):
    st.warning("Add GROQ_API_KEY to the .env file before asking a question.")
    st.stop()

question = st.text_area(
    "Question",
    placeholder="What did the document say about inflation?",
    height=100,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Searching the document and generating an answer..."):
            chain = build_chain(str(selected_path), "openai/gpt-oss-120b")
            answer = chain.invoke(question.strip())
        st.subheader("Answer")
        st.write(answer)
    except Exception as error:
        st.error(f"Unable to answer the question: {error}")