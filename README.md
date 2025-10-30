# 📘 PaperIQ — Smart Document Intelligence Assistant

PaperIQ is a multi-document AI assistant that lets you upload PDFs and text files, ask natural-language questions across all documents, and generate combined summaries.

## Features
- Upload multiple PDFs / TXT files
- Vector search powered Q&A (OpenAI embeddings + Chroma)
- Returns source snippets for answers
- One-click "Summarize All Documents" with structured summary
- Save/download answers and summaries

## Tech Stack
- LangChain (chaining and document tooling)
- OpenAI (embeddings + Chat models)
- ChromaDB (vector store)
- Streamlit (UI)

## Quickstart (run locally)
1. Clone the repo:
```bash
git clone https://github.com/<your-username>/PaperIQ.git
cd PaperIQ
````
2.Install dependencies:
```bash
pip install -r requirements.txt
```

3.Set your OpenAI API key:

macOS / Linux:
```bash
export OPENAI_API_KEY="sk-..."
```

Windows PowerShell:
```bash
$env:OPENAI_API_KEY="sk-..."
```

Run the app:
```bash
streamlit run paperiq_app.py
```
Upload PDFs and test with a question or click "Summarize All Documents".
