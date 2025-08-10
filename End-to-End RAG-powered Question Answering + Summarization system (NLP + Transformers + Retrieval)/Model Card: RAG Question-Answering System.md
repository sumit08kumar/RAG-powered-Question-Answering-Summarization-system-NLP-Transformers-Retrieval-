# Model Card: RAG Question-Answering System

## Model Details

**Model Name:** RAG Question-Answering + Summarization System  
**Version:** 1.0.0  
**Date:** August 2025  
**Model Type:** Retrieval-Augmented Generation (RAG)  

### Architecture Components

1. **Embedding Model:** `all-mpnet-base-v2` (sentence-transformers)
   - Purpose: Convert text chunks into dense vector representations
   - Dimension: 768
   - Performance: Optimized for semantic similarity tasks

2. **Vector Store:** FAISS (Facebook AI Similarity Search)
   - Purpose: Efficient similarity search and clustering of dense vectors
   - Index Type: HNSW (Hierarchical Navigable Small World) for large datasets, Flat for small datasets
   - Distance Metric: Inner Product (cosine similarity)

3. **Generation Model:** `google/flan-t5-base`
   - Purpose: Generate answers and summaries based on retrieved context
   - Parameters: 248M
   - Architecture: Text-to-Text Transfer Transformer (T5)

4. **Orchestration:** LangChain
   - Purpose: Manage retrieval and generation pipeline
   - Features: Prompt templates, chain composition, document processing

## Intended Use

### Primary Use Cases
- Document question-answering for knowledge bases
- Summarization of document collections
- Information retrieval from uploaded documents

### Intended Users
- Researchers and analysts
- Knowledge workers
- Students and educators
- Organizations with document repositories

## Performance Characteristics

### Capabilities
- Processes PDF, TXT, and Markdown documents
- Semantic chunking with configurable overlap
- Retrieval of top-k relevant document chunks
- Context-aware answer generation
- Source attribution and citation
- Multi-document summarization

### Limitations
- Limited to text-based documents
- Performance depends on document quality and structure
- Generation quality varies with query complexity
- May hallucinate information not present in documents
- Processing time increases with document collection size

## Training Data

### Embedding Model
- Trained on diverse text corpora including web pages, books, and academic papers
- Optimized for semantic similarity tasks
- No additional fine-tuning performed

### Generation Model
- Pre-trained on web text and fine-tuned on instruction-following tasks
- Supports multiple languages but optimized for English
- No domain-specific fine-tuning performed

## Evaluation

### Metrics
- **Retrieval Recall@k:** Measures ability to retrieve relevant documents
- **Answer Relevance:** Human evaluation of answer quality and relevance
- **Source Attribution:** Accuracy of source citations
- **Response Time:** End-to-end latency for queries

### Known Issues
- May struggle with highly technical or domain-specific terminology
- Performance degrades with very long documents
- Occasional inconsistencies in citation formatting

## Ethical Considerations

### Bias and Fairness
- Inherits biases from pre-trained models
- Performance may vary across different domains and languages
- No specific bias mitigation techniques implemented

### Privacy and Security
- Documents are processed locally within the system
- No data is sent to external services during inference
- Users should ensure sensitive documents are handled appropriately

## Deployment and Usage

### System Requirements
- Python 3.11+
- 4GB+ RAM (8GB+ recommended)
- CPU-based inference (GPU optional)

### API Endpoints
- `POST /ingest` - Upload and process documents
- `POST /query` - Ask questions about documents
- `POST /summarize` - Generate document summaries
- `GET /health` - System status check

## Maintenance and Updates

### Model Updates
- Embedding and generation models can be updated independently
- Vector indices may need rebuilding after model updates
- Configuration allows for easy model swapping

### Monitoring
- Track query response times
- Monitor answer quality through user feedback
- Log retrieval performance metrics

## Contact and Support

For questions, issues, or contributions, please refer to the project documentation and repository.

## License

This system uses models and libraries with various licenses:
- sentence-transformers: Apache 2.0
- FAISS: MIT
- Transformers: Apache 2.0
- LangChain: MIT
- FastAPI: MIT
- Streamlit: Apache 2.0

Please review individual component licenses for specific terms and conditions.

