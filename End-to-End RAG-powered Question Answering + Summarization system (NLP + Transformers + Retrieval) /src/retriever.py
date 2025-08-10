"""
Retrieval module for RAG system.
Uses LangChain when available; otherwise falls back to a lightweight retriever
built on sentence-transformers + FAISS/NumPy from embed.py.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try LangChain stack
LANGCHAIN_AVAILABLE = True
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline
except Exception:
    print("Warning: LangChain and related packages not yet installed")
    LANGCHAIN_AVAILABLE = False
    FAISS = None  # type: ignore
    HuggingFaceEmbeddings = None  # type: ignore
    Document = None  # type: ignore

# Minimal Document fallback for type-compat
if Document is None:
    class Document:  # type: ignore
        def __init__(self, page_content: str, metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Fallback embedding/index manager
try:
    from embed import EmbeddingManager
except Exception as e:
    EmbeddingManager = None  # type: ignore
    print(f"Warning: embed.py not available for fallback retriever: {e}")


# ---------------------- LangChain implementation ---------------------- #

class LangChainRetriever:
    """LangChain-based retriever for RAG system."""

    def __init__(self, embedding_model: str = 'all-mpnet-base-v2', vector_store_path: str = None):
        self.embedding_model_name = embedding_model
        self.vector_store_path = vector_store_path
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.compression_retriever = None

        if HuggingFaceEmbeddings is not None:
            self._initialize_embeddings()

    def _initialize_embeddings(self):
        try:
            logger.info(f"Initializing embeddings with model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def create_vector_store_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not initialized")
        if not chunks:
            logger.warning("No chunks provided to create vector store")
            return

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'chunk_id': chunk.get('chunk_id', 0),
                    'source_document': chunk.get('source_document', 'unknown'),
                    'start_pos': chunk.get('start_pos', 0),
                    'end_pos': chunk.get('end_pos', 0),
                    **chunk.get('document_metadata', {}),
                },
            )
            documents.append(doc)

        logger.info(f"Creating FAISS vector store from {len(documents)} documents")
        self.vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        logger.info("Vector store created successfully")

    def save_vector_store(self, path: str = None):
        if self.vector_store is None:
            raise RuntimeError("No vector store to save")
        save_path = path or self.vector_store_path or "langchain_faiss_store"
        self.vector_store.save_local(save_path)
        logger.info(f"Vector store saved to {save_path}")

    def load_vector_store(self, path: str = None):
        if self.embeddings is None:
            raise RuntimeError("Embeddings not initialized")
        load_path = path or self.vector_store_path or "langchain_faiss_store"
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        self.vector_store = FAISS.load_local(load_path, self.embeddings)
        logger.info(f"Vector store loaded from {load_path}")

    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        self.retriever = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
        return self.retriever

    def get_compression_retriever(self, llm_model: str = "google/flan-t5-base", k: int = 10, compressed_k: int = 5):
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        try:
            base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            hf_pipeline = pipeline("text2text-generation", model=llm_model, max_length=512, temperature=0.1)
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            compressor = LLMChainExtractor.from_llm(llm)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )
            return self.compression_retriever
        except Exception as e:
            logger.warning(f"Failed to create compression retriever: {e}")
            return self.get_retriever(k=compressed_k)

    def retrieve_documents(self, query: str, use_compression: bool = False) -> List[Document]:
        if use_compression and self.compression_retriever is not None:
            retriever = self.compression_retriever
        elif self.retriever is not None:
            retriever = self.retriever
        else:
            retriever = self.get_retriever()
        try:
            documents = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def search_similarity(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [
                {'rank': i + 1, 'score': float(score), 'content': doc.page_content, 'metadata': doc.metadata}
                for i, (doc, score) in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []


# ---------------------- Lightweight fallback implementation ---------------------- #

class SimpleRetriever:
    """
    Lightweight retriever using EmbeddingManager (sentence-transformers + FAISS/NumPy).
    Provides the same public methods used by app.py.
    """

    def __init__(self, embedding_model: str = 'all-mpnet-base-v2', vector_store_path: str = None):
        if EmbeddingManager is None:
            raise RuntimeError("embed.py not available. Install sentence-transformers or provide EmbeddingManager.")
        index_prefix = str(Path(vector_store_path) / "faiss_index") if vector_store_path else "faiss_index"
        self.em = EmbeddingManager(model_name=embedding_model, index_path=index_prefix)
        self.k_default = 5

    def create_vector_store_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            logger.warning("No chunks provided to create vector store")
            return
        try:
            self.em.add_chunks_to_index(chunks)
            logger.info("Vector store created with fallback retriever")
        except Exception as e:
            logger.error(f"Failed to build fallback index: {e}")
            raise

    def save_vector_store(self, path: str = None):
        prefix = str(Path(path) / "faiss_index") if path else None
        self.em.save_index(prefix)

    def load_vector_store(self, path: str = None):
        prefix = str(Path(path) / "faiss_index") if path else None
        self.em.load_index(prefix)

    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        # No-op for API parity; returns self
        self.k_default = k
        return self

    def retrieve_documents(self, query: str, use_compression: bool = False) -> List[Document]:
        results = self.em.search(query, k=self.k_default)
        docs: List[Document] = []
        for r in results:
            meta = r.get("chunk_metadata", {}) or {}
            meta["score"] = r.get("score", 0.0)
            docs.append(Document(page_content=r.get("content", ""), metadata=meta))
        return docs

    def search_similarity(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = self.em.search(query, k=k)
        formatted: List[Dict[str, Any]] = []
        for i, r in enumerate(results):
            formatted.append({
                "rank": i + 1,
                "score": float(r.get("score", 0.0)),
                "content": r.get("content", ""),
                "metadata": r.get("chunk_metadata", {}) or {},
            })
        return formatted


def create_retriever_from_chunks(
    chunks: List[Dict[str, Any]],
    embedding_model: str = 'all-mpnet-base-v2',
    save_path: str = None,
):
    """
    Create a retriever from document chunks.
    Uses LangChain when available; otherwise uses SimpleRetriever.
    """
    if LANGCHAIN_AVAILABLE:
        retr = LangChainRetriever(embedding_model=embedding_model, vector_store_path=save_path)
        retr.create_vector_store_from_chunks(chunks)
        if save_path:
            retr.save_vector_store(save_path)
        return retr

    # Fallback
    retr = SimpleRetriever(embedding_model=embedding_model, vector_store_path=save_path)
    retr.create_vector_store_from_chunks(chunks)
    if save_path:
        retr.save_vector_store(save_path)
    return retr


if __name__ == "__main__":
    # Simple smoke test for both paths
    logging.basicConfig(level=logging.INFO)

    sample_chunks = [
        {'content': 'Machine learning is a subset of AI.', 'chunk_id': 0, 'source_document': 'ml.txt'},
        {'content': 'Deep learning uses neural networks.', 'chunk_id': 1, 'source_document': 'dl.txt'},
        {'content': 'NLP lets computers understand language.', 'chunk_id': 2, 'source_document': 'nlp.txt'},
    ]

    try:
        retriever = create_retriever_from_chunks(sample_chunks)
        print(retriever.search_similarity("What is machine learning?", k=2))
    except Exception as e:
        print(f"Error (expected if dependencies not installed): {e}")