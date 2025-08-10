"""
Streamlit frontend for RAG Question-Answering System.
Provides a user-friendly interface for document upload and querying.
"""

import os
import time
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def check_api_health():
    """Check if the API is running and healthy."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200, r.json() if r.headers.get("content-type","").startswith("application/json") else None
    except requests.exceptions.RequestException:
        return False, None

def upload_documents(files):
    """Upload documents to the API (sync by default)."""
    try:
        files_data = []
        for f in files:
            content_type = f.type or "application/octet-stream"
            files_data.append(("files", (f.name, f.getvalue(), content_type)))
        r = requests.post(
            f"{API_BASE_URL}/ingest",
            params={"async_mode": "false"},  # force sync so index_ready flips now
            files=files_data,
            timeout=300,
        )
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code in (200, 201), r.json()
        return r.status_code in (200, 201), r.text
    except requests.exceptions.RequestException as e:
        return False, str(e)

def query_documents(question: str, max_results: int = 5, use_compression: bool = False):
    try:
        payload = {"question": question, "max_results": max_results, "use_compression": use_compression}
        r = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code == 200, r.json()
        return r.status_code == 200, r.text
    except requests.exceptions.RequestException as e:
        return False, str(e)

def summarize_documents(query: str = None, max_documents: int = 10):
    try:
        payload: Dict[str, Any] = {"max_documents": max_documents}
        if query:
            payload["query"] = query
        r = requests.post(f"{API_BASE_URL}/summarize", json=payload, timeout=60)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code == 200, r.json()
        return r.status_code == 200, r.text
    except requests.exceptions.RequestException as e:
        return False, str(e)

def main():
    st.title("🤖 RAG Question-Answering System")
    st.markdown("Upload documents and ask questions to get AI-powered answers with source citations.")

    ok, health_data = check_api_health()
    if not ok:
        st.error("⚠️ Backend API is not running.")
        parsed = urlparse(API_BASE_URL)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        st.markdown(f"- Expected API at: {API_BASE_URL}")
        st.code(f"uvicorn src.app:app --host 0.0.0.0 --port {port} --reload")
        st.stop()

    # Sidebar with live-refresh capability
    sidebar_placeholder = st.sidebar.empty()
    def render_sidebar(hd: Dict[str, Any]):
        with sidebar_placeholder.container():
            st.header("📊 System Status")
            st.success("✅ API Connected")
            st.write(f"API Base URL: `{API_BASE_URL}`")
            st.write(f"**Status:** {hd.get('status', 'Unknown')}")
            st.write(f"**Documents Loaded:** {hd.get('documents_loaded', 0)}")
            st.write(f"**Index Ready:** {'✅' if hd.get('index_ready') else '❌'}")
            if hd.get("last_error"):
                st.error(f"Last error: {hd.get('last_error')}")
            if msg := hd.get("message"):
                st.caption(msg)

    render_sidebar(health_data)

    tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "❓ Ask Questions", "📝 Summarize"])

    # Upload tab
    with tab1:
        st.header("📁 Document Upload")
        st.markdown("Upload PDF, TXT, or Markdown files to build your knowledge base.")
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "md"], accept_multiple_files=True)

        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for f in uploaded_files:
                st.write(f"- {f.name} ({f.size} bytes)")

            if st.button("🚀 Upload and Process", type="primary"):
                with st.spinner("Uploading and processing documents..."):
                    success, result = upload_documents(uploaded_files)
                    if success:
                        st.success("✅ Upload finished.")
                        try:
                            st.json(result)
                        except Exception:
                            st.write(result)

                        ok, current_health = check_api_health()
                        if ok and current_health:
                            render_sidebar(current_health)
                            if current_health.get("index_ready"):
                                st.success("✅ Documents processed and indexed!")
                                st.rerun()
                        if not current_health or not current_health.get("index_ready"):
                            # quick fallback poll
                            progress_placeholder = st.empty()
                            for i in range(15):
                                time.sleep(2)
                                ok, current_health = check_api_health()
                                if ok and current_health:
                                    render_sidebar(current_health)
                                    if current_health.get("index_ready"):
                                        progress_placeholder.success("✅ Documents processed and indexed!")
                                        st.rerun()
                                        break
                                    if current_health.get("last_error"):
                                        progress_placeholder.error(f"❌ Error: {current_health['last_error']}")
                                        break
                                progress_placeholder.info(f"⏳ Finalizing... ({(i + 1) * 2}s)")
                    else:
                        st.error(f"❌ Upload failed: {result}")

    # Q&A tab
    with tab2:
        st.header("❓ Ask Questions")
        ok, health_data = check_api_health()
        if not ok or not health_data or not health_data.get("index_ready"):
            st.warning("⚠️ Please upload documents first before asking questions.")
        else:
            st.markdown("Ask questions about your uploaded documents and get answers with source citations.")
            question = st.text_area("Enter your question:", placeholder="What is machine learning?", height=100)
            with st.expander("⚙️ Advanced Options"):
                max_results = st.slider("Maximum results", 1, 10, 5)
                use_compression = st.checkbox("Use compression retriever", help="May improve relevance but slower")

            if st.button("🔍 Get Answer", type="primary", disabled=not question.strip()):
                with st.spinner("Searching documents and generating answer..."):
                    success, result = query_documents(question, max_results, use_compression)
                    if success and isinstance(result, dict):
                        st.success("✅ Answer generated!")
                        st.subheader("📝 Answer")
                        st.write(result.get("answer", ""))
                        confidence = float(result.get("confidence", 0.0) or 0.0)
                        st.metric("Confidence", f"{confidence:.1%}")
                        if result.get("sources"):
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result["sources"], 1):
                                meta = source.get("metadata", {}) or {}
                                title = meta.get("source_document") or meta.get("source") or "Unknown"
                                with st.expander(f"Source {i}: {title}"):
                                    st.write(source.get("content", ""))
                                    try:
                                        st.json(meta)
                                    except Exception:
                                        st.write(meta)
                        if result.get("context_used"):
                            with st.expander("🔍 Context Used"):
                                st.text(result["context_used"])
                    else:
                        st.error(f"❌ Query failed: {result}")

    # Summarize tab
    with tab3:
        st.header("📝 Document Summarization")
        ok, health_data = check_api_health()
        if not ok or not health_data or not health_data.get("index_ready"):
            st.warning("⚠️ Please upload documents first before generating summaries.")
        else:
            st.markdown("Generate summaries of your document collection.")
            summary_query = st.text_input("Optional: Filter by topic", placeholder="machine learning")
            max_documents = st.slider("Maximum documents to summarize", 1, 20, 10)
            if st.button("📝 Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    query = summary_query.strip() if summary_query and summary_query.strip() else None
                    success, result = summarize_documents(query, max_documents)
                    if success and isinstance(result, dict):
                        st.success("✅ Summary generated!")
                        st.subheader("📄 Summary")
                        st.write(result.get("summary", ""))
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Documents Processed", result.get("num_documents", 0))
                        with col2:
                            st.metric("Input Length", f"{result.get('input_length', 0)} chars")
                    else:
                        st.error(f"❌ Summarization failed: {result}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Built with ❤️ using FastAPI, LangChain, FAISS, and Streamlit</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()