
import os

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

st.set_page_config(
    page_title="Simple RAG Chat",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# LOAD EMBEDDING MODEL ONLY ONCE
# --------------------------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# BUILD RETRIEVER AND FAISS VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource(show_spinner="Building the document index...")

def build_retriever(document_text: str, filename: str):

    # Convert text into a LangChain Document
    document = Document(
        page_content=document_text,
        metadata={"source": filename}
    )

    # Split document into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks = text_splitter.split_documents([document])

    # Reuse the cached embedding model
    embeddings = load_embeddings()

    # Build FAISS vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Return retriever
    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


# --------------------------------------------------
# BUILD RAG CHAIN
# --------------------------------------------------

@st.cache_resource
def build_chain(document_text: str, filename: str, model_name: str):

    retriever = build_retriever(
        document_text,
        filename
    )

    template = """Answer the question based only on the following context:

{context}

Question: {question}

If the answer is not in the context, say that the document does not provide enough information.
"""

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatGroq(
        model=model_name,
        temperature=0,
        streaming=True
    )

    def format_docs(docs):

        return "\n\n".join(
            document.page_content for document in docs
        )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )


# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("Simple RAG Chat")

st.caption(
    "Upload your own text documents and ask questions about their content."
)


# --------------------------------------------------
# UPLOAD DOCUMENTS
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["txt"],
    accept_multiple_files=True,
    help="You can upload multiple .txt files."
)


if not uploaded_files:

    st.info("Please upload at least one .txt document to get started.")

    st.stop()


st.success(
    f"{len(uploaded_files)} document(s) uploaded successfully!"
)


# --------------------------------------------------
# SELECT DOCUMENT
# --------------------------------------------------

selected_file = st.selectbox(
    "Select a document",
    uploaded_files,
    format_func=lambda file: file.name
)


# --------------------------------------------------
# READ DOCUMENT
# --------------------------------------------------

try:

    document_text = selected_file.getvalue().decode("utf-8")

except UnicodeDecodeError:

    st.error(
        "Unable to read this file. Please upload a UTF-8 encoded text file."
    )

    st.stop()


if not document_text.strip():

    st.error("The uploaded document is empty.")

    st.stop()


# --------------------------------------------------
# CHECK GROQ API KEY
# --------------------------------------------------

if not os.getenv("GROQ_API_KEY"):

    st.warning(
        "Please configure your GROQ_API_KEY in the environment variables."
    )

    st.stop()


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

question = st.text_area(
    "Ask a question about your document",
    placeholder="What did the document say about inflation?",
    height=100
)


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

if st.button(
    "Ask",
    type="primary",
    disabled=not question.strip()
):

    try:

        chain = build_chain(
            document_text,
            selected_file.name,
            "openai/gpt-oss-120b"
        )

        st.subheader("Answer")

        with st.spinner("Searching your document..."):

            response = chain.stream(question.strip())

            st.write_stream(response)

    except Exception as error:

        st.error(
            f"Unable to answer the question: {error}"
        )
