import os
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import ChatPromptTemplate


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="😊",
    layout="wide"
)


# ---------------- HEADER ----------------
st.title("😊 Multi-Document RAG Chatbot")

st.markdown("""
Ask questions about your uploaded knowledge base.

### Supported Files
- PDF
- DOCX
- TXT
- CSV
- XLSX
""")


# ---------------- DOCUMENT LOADER ----------------
def load_document(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")

    elif extension == ".csv":
        loader = CSVLoader(file_path)

    elif extension in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        return []

    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = os.path.basename(file_path)

    return docs


# ---------------- LOAD RAG ----------------
@st.cache_resource
def build_rag():

    folder_path = "knowledge_base"

    all_docs = []

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):

            try:
                docs = load_document(file_path)

                if docs:
                    all_docs.extend(docs)

            except:
                pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":5}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key="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"
    )

    system_prompt = """
You are a helpful AI assistant.

Answer ONLY using the provided context whenever possible.

If the answer is not found, clearly say it is not present in the uploaded documents and then provide your best general knowledge answer.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        qa_chain
    )

    return rag_chain, len(all_docs), len(split_docs)


rag_chain, total_docs, total_chunks = build_rag()


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("Knowledge Base")

    st.metric("Documents", total_docs)
    st.metric("Chunks", total_chunks)

    st.success("RAG Ready")


# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display old messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- USER INPUT ----------------
question = st.chat_input("Ask a question...")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = rag_chain.invoke(
                {
                    "input":question
                }
            )

            answer = response["answer"]

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )       