"""
Embedding and vector indexing module for RAG system.
Uses sentence-transformers for embeddings and FAISS for vector storage.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

# Import will be available after installation completes
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Warning: sentence-transformers and faiss-cpu not yet installed")
    SentenceTransformer = None
    faiss = None

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings and FAISS vector index."""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', index_path: str = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            index_path: Path to save/load the FAISS index
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.index_path = index_path or "faiss_index"
        self.metadata_path = f"{self.index_path}_metadata.json"
        self.chunks_metadata = []
        self.dimension = None
        
        # Initialize model if available
        if SentenceTransformer is not None:
            self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            sample_embedding = self.model.encode(["test"])
            self.dimension = sample_embedding.shape[1]
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Install sentence-transformers first.")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings.astype('float32')
    
    def create_index(self, embeddings: np.ndarray, use_gpu: bool = False):
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings
            use_gpu: Whether to use GPU for indexing (if available)
        """
        if faiss is None:
            raise RuntimeError("FAISS not available. Install faiss-cpu first.")
        
        dimension = embeddings.shape[1]
        n_embeddings = embeddings.shape[0]
        
        logger.info(f"Creating FAISS index for {n_embeddings} embeddings of dimension {dimension}")
        
        # Choose index type based on dataset size
        if n_embeddings < 10000:
            # Use flat index for small datasets (exact search)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        else:
            # Use HNSW for larger datasets (approximate search)
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        logger.info(f"Index created with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: str = None):
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        save_path = index_path or self.index_path
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}.faiss")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'num_vectors': self.index.ntotal,
            'chunks_metadata': self.chunks_metadata
        }
        
        with open(f"{save_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Index and metadata saved to {save_path}")
    
    def load_index(self, index_path: str = None):
        """Load a FAISS index and metadata from disk."""
        if faiss is None:
            raise RuntimeError("FAISS not available. Install faiss-cpu first.")
        
        load_path = index_path or self.index_path
        
        # Load FAISS index
        index_file = f"{load_path}.faiss"
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        metadata_file = f"{load_path}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get('model_name', self.model_name)
            self.dimension = metadata.get('dimension')
            self.chunks_metadata = metadata.get('chunks_metadata', [])
        
        logger.info(f"Index loaded with {self.index.ntotal} vectors")
    
    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar texts in the index.
        
        Args:
            query_text: Query text to search for
            k: Number of top results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if self.model is None or self.index is None:
            raise RuntimeError("Model and index must be loaded first")
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text])
        query_embedding = query_embedding.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks_metadata):
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'chunk_metadata': self.chunks_metadata[idx],
                    'content': self.chunks_metadata[idx].get('content', '')
                }
                results.append(result)
        
        return results
    
    def add_chunks_to_index(self, chunks: List[Dict[str, Any]]):
        """
        Add text chunks to the index.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and metadata
        """
        # Extract text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Store metadata
        self.chunks_metadata.extend(chunks)
        
        # Create or update index
        if self.index is None:
            self.create_index(embeddings)
        else:
            # Add to existing index
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        
        logger.info(f"Added {len(chunks)} chunks to index. Total: {self.index.ntotal}")


def build_index_from_chunks(chunks: List[Dict[str, Any]], 
                          model_name: str = 'all-mpnet-base-v2',
                          index_path: str = "faiss_index") -> EmbeddingManager:
    """
    Build a FAISS index from document chunks.
    
    Args:
        chunks: List of document chunks with content and metadata
        model_name: Sentence transformer model name
        index_path: Path to save the index
        
    Returns:
        EmbeddingManager instance with built index
    """
    embedding_manager = EmbeddingManager(model_name=model_name, index_path=index_path)
    
    if not chunks:
        logger.warning("No chunks provided to build index")
        return embedding_manager
    
    # Add chunks to index
    embedding_manager.add_chunks_to_index(chunks)
    
    # Save index
    embedding_manager.save_index()
    
    return embedding_manager


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample chunks for testing
    sample_chunks = [
        {
            'content': 'This is a sample document about machine learning.',
            'chunk_id': 0,
            'source_document': 'sample.txt'
        },
        {
            'content': 'Natural language processing is a subfield of AI.',
            'chunk_id': 1,
            'source_document': 'sample.txt'
        }
    ]
    
    try:
        # Build index
        em = build_index_from_chunks(sample_chunks)
        
        # Test search
        results = em.search("machine learning", k=2)
        
        print("Search results:")
        for result in results:
            print(f"Score: {result['score']:.3f}")
            print(f"Content: {result['content']}")
            print("---")
            
    except Exception as e:
        print(f"Error (expected if dependencies not installed): {e}")

