# paperiq_app.py
"""
PaperIQ - Smart Document Intelligence Assistant (Streamlit app)
Features:
 - Multi-PDF / TXT upload
 - Chunking + embeddings (OpenAI embeddings)
 - Chroma vector store (optional persistent dir)
 - RetrievalQA for question answering (with source snippets)
 - "Summarize All Documents" button (structured summary)
 - Save answer / summary to a text file for download
Requirements:
 - Set OPENAI_API_KEY in environment before running.
"""

import os
import tempfile
import uuid
from typing import List, Tuple

import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ---------- CONFIG ----------
APP_TITLE = "📘 PaperIQ — Smart Document Intelligence Assistant"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.2
PERSIST_DIR = os.path.join(tempfile.gettempdir(), "paperiq_chroma")  # change if you want persistent store
# ----------------------------

st.set_page_config(page_title="PaperIQ", layout="wide")
st.title(APP_TITLE)
st.markdown(
    """
Upload one or more PDFs / text files. Ask questions across all uploaded documents and generate a combined summary.
"""
)

# Sidebar - settings
with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (characters)", min_value=200, max_value=5000, value=DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=1000, value=DEFAULT_CHUNK_OVERLAP, step=10)
    top_k = st.number_input("Retriever top_k", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)
    persist_store = st.checkbox("Use persistent vector store (persist across runs)", value=False)
    if persist_store:
        persist_dir = st.text_input("Chroma persist directory", value=PERSIST_DIR)
    else:
        persist_dir = None

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OpenAI API key not set. Set environment variable OPENAI_API_KEY before running.")
    st.stop()

# File upload
uploaded_files = st.file_uploader("Upload PDFs or text files (one or more)", accept_multiple_files=True, type=["pdf", "txt"])

if not uploaded_files:
    st.info("Upload files to get started. You can upload multiple PDFs or text files.")
    st.stop()

# Helper functions
def load_uploaded_files(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        suffix = f.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()
                docs.extend(loaded)
            else:
                loader = TextLoader(tmp_path, encoding="utf8")
                loaded = loader.load()
                docs.extend(loaded)
        except Exception as e:
            st.warning(f"Failed to load {f.name}: {e}")
    return docs

def build_vector_store(documents: List[Document], embeddings_model, persist_directory: str = None) -> Chroma:
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were produced from the uploaded documents.")
    # Create vector store
    if persist_directory:
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=persist_directory)
        vectordb.persist()
    else:
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings_model)
    return vectordb

def make_qa_chain(vectordb: Chroma, temp: float) -> RetrievalQA:
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    llm = ChatOpenAI(model="gpt-4", temperature=temp, max_tokens=800)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def get_source_snippets(source_docs: List[Document]) -> str:
    snippets = []
    for i, sd in enumerate(source_docs):
        # include filename or metadata if available
        src_meta = sd.metadata.get("source", f"chunk_{i+1}")
        text = sd.page_content.strip()
        if len(text) > 1000:
            text = text[:1000] + "..."
        snippets.append(f"--- Source {i+1}: {src_meta} ---\n{text}\n")
    return "\n".join(snippets)

# Process upload & build DB
with st.spinner("Loading documents..."):
    documents = load_uploaded_files(uploaded_files)
    if not documents:
        st.error("No text found in uploaded files.")
        st.stop()

st.success(f"Loaded {len(documents)} document pages/entries.")
st.write("Files uploaded:", ", ".join([f.name for f in uploaded_files]))

# Build embeddings & vector store
with st.spinner("Building embeddings and vector store (this may take a moment)..."):
    embeddings = OpenAIEmbeddings()
    try:
        vectordb = build_vector_store(documents, embeddings, persist_directory=persist_dir if persist_store else None)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()

# Create QA chain
qa_chain = make_qa_chain(vectordb, temperature)

# Layout: left - QA, right - Summary & options
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a question")
    query = st.text_input("Type your question about the uploaded documents here:")
    if st.button("Get Answer") and query.strip():
        with st.spinner("Querying..."):
            try:
                res = qa_chain({"query": query})
                answer = res.get("result") or res.get("answer") or ""
                sources = res.get("source_documents", [])
            except Exception as e:
                st.error(f"Error during query: {e}")
                answer = ""
                sources = []

        if answer:
            st.markdown("### ✅ Answer")
            st.write(answer)
            # Sources
            if sources:
                st.markdown("#### Source snippets (top results)")
                for i, sd in enumerate(sources[:min(len(sources), top_k)]):
                    src_meta = sd.metadata.get("source", f"chunk_{i+1}")
                    st.markdown(f"**Source {i+1} — {src_meta}**")
                    st.write(sd.page_content[:1000] + ("..." if len(sd.page_content) > 1000 else ""))
                    st.write("---")

            # Save answer to downloadable file
            if st.button("Save Answer to file"):
                out_text = f"Question:\n{query}\n\nAnswer:\n{answer}\n\n--- Sources ---\n{get_source_snippets(sources)}"
                out_path = os.path.join(tempfile.gettempdir(), f"paperiq_answer_{uuid.uuid4().hex[:8]}.txt")
                with open(out_path, "w", encoding="utf8") as f:
                    f.write(out_text)
                with open(out_path, "rb") as f:
                    st.download_button("Download Answer File", data=f, file_name=os.path.basename(out_path), mime="text/plain")

with col2:
    st.subheader("Summary & Utilities")
    if st.button("🧾 Summarize All Documents"):
        with st.spinner("Generating combined summary..."):
            # Use retrieval to get top chunks then ask LLM to summarize combined content
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k * 2})
            top_chunks = retriever.get_relevant_documents("summarize all documents")
            combined_text = "\n\n".join([c.page_content for c in top_chunks])
            # Use a direct chat call for summarization
            llm = ChatOpenAI(model="gpt-4", temperature=0.1, max_tokens=1000)
            prompt = (
                "You are an AI assistant. Generate a concise structured summary of the following documents."
                " Include: Short overall summary, Key points, Actionable insights, Potential limitations/uncertainties.\n\n"
                f"DOCUMENTS:\n{combined_text}"
            )
            try:
                summary = llm.predict(prompt)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                summary = ""

        if summary:
            st.markdown("### 📄 Combined Summary")
            st.write(summary)
            if st.button("Download Summary"):
                out_path = os.path.join(tempfile.gettempdir(), f"paperiq_summary_{uuid.uuid4().hex[:8]}.txt")
                with open(out_path, "w", encoding="utf8") as f:
                    f.write(summary)
                with open(out_path, "rb") as f:
                    st.download_button("Download Summary File", data=f, file_name=os.path.basename(out_path), mime="text/plain")

    st.markdown("### Options")
    if st.button("Clear Vector Store (local memory)"):
        try:
            # delete persistent folder if using persist
            if persist_store and persist_dir:
                if os.path.exists(persist_dir):
                    import shutil
                    shutil.rmtree(persist_dir)
                    st.success("Persistent vector store cleared.")
                else:
                    st.info("No persistent store found.")
            else:
                # Recreate an in-memory empty chroma by reinitializing vectordb to empty
                vectordb = Chroma.from_documents([], embedding=embeddings)
                st.success("In-memory vector store reset for this session.")
        except Exception as e:
            st.error(f"Failed to clear vector store: {e}")

st.markdown("---")
st.markdown("Built with LangChain + OpenAI + Chroma. Keep temperature low for factual answers.")
