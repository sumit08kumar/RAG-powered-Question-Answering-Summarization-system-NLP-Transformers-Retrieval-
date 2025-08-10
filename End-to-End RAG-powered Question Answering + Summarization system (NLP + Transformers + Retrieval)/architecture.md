# Architecture Diagram

This document will contain detailed architecture diagrams for the RAG system, covering:

1.  **Ingestion & Preprocessing**: How documents are converted, cleaned, and chunked.
2.  **Embeddings & Indexing**: How text chunks are transformed into embeddings and stored in FAISS.
3.  **Retrieval**: The process of searching and retrieving relevant chunks based on a user query.
4.  **Context Assembly & Generation**: How retrieved chunks are used to construct prompts for the LLM and generate answers/summaries.
5.  **API & UI**: The interaction between the FastAPI backend and the Gradio/Streamlit frontend.

## High-level Flow

User Query -> FastAPI -> LangChain (Retriever -> Reranker -> Generator) -> LLM -> Response

## Component Breakdown

- **Ingestion**: `pdfplumber`, `pypdf`, `boilerpy3`, `newspaper3k`
- **Embeddings**: `sentence-transformers`
- **Vector DB**: `FAISS`
- **Orchestration**: `LangChain`
- **Backend API**: `FastAPI`
- **Frontend UI**: `Gradio` / `Streamlit`
- **LLM**: Hugging Face models (e.g., T5, BART, Flan-T5)


