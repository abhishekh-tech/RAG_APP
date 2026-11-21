
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


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

st.set_page_config(
    page_title="Simple RAG Chat",
    page_icon="📚",
    layout="centered"
)


# Build FAISS retriever from uploaded document
@st.cache_resource(show_spinner="Building the document index...")
def build_retriever(document_text: str, filename: str):

    # Convert uploaded text into LangChain Document
    document = Document(
        page_content=document_text,
        metadata={"source": filename}
    )

    # Split document into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks_of_text = text_splitter.split_documents([document])

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    vector_db = FAISS.from_documents(
        chunks_of_text,
        embeddings
    )

    return vector_db.as_retriever(
        search_kwargs={"k": 3}
    )


# Build RAG chain
@st.cache_resource
def build_chain(document_text: str, filename: str, model_name: str):

    retriever = build_retriever(document_text, filename)

    template = """Answer the question based only on the following context:

{context}

Question: {question}

If the answer is not in the context, say that the document does not provide enough information.
"""

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatGroq(
        model=model_name,
        temperature=0
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


# Application UI
st.title("Simple RAG Chat")

st.caption(
    "Upload your own text documents and ask questions about their content."
)


# Upload documents from computer
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["txt"],
    accept_multiple_files=True,
    help="You can upload multiple .txt files."
)


# Check if documents are uploaded
if not uploaded_files:
    st.info("Please upload at least one .txt document to get started.")
    st.stop()


# Display uploaded documents
st.success(f"{len(uploaded_files)} document(s) uploaded successfully!")


# Allow users to select a document
selected_file = st.selectbox(
    "Select a document",
    uploaded_files,
    format_func=lambda file: file.name
)


# Read uploaded document content
try:
    document_text = selected_file.getvalue().decode("utf-8")

except UnicodeDecodeError:
    st.error("Unable to read this file. Please upload a UTF-8 encoded text file.")
    st.stop()


if not document_text.strip():
    st.error("The uploaded document is empty.")
    st.stop()


# Check Groq API key
if not os.getenv("GROQ_API_KEY"):
    st.warning("Please configure your GROQ_API_KEY in the environment variables.")
    st.stop()


# Question input
question = st.text_area(
    "Ask a question about your document",
    placeholder="What did the document say about inflation?",
    height=100
)


# Generate answer
if st.button(
    "Ask",
    type="primary",
    disabled=not question.strip()
):

    try:
        with st.spinner(
            "Searching the document and generating an answer..."
        ):

            chain = build_chain(
                document_text,
                selected_file.name,
                "openai/gpt-oss-120b"
            )

            answer = chain.invoke(question.strip())

        st.subheader("Answer")
        st.write(answer)

    except Exception as error:
        st.error(f"Unable to answer the question: {error}")
