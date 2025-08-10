# 🤖 RAG Question-Answering + Summarization System

A production-ready **Retrieval-Augmented Generation (RAG)** system for document question-answering and summarization, built with FastAPI, LangChain, FAISS, and Streamlit.

## 🚀 Features

- **Document Upload & Processing**: Support for PDF, TXT, and Markdown files
- **Intelligent Chunking**: Semantic text chunking with configurable overlap
- **Vector Search**: FAISS-powered similarity search with sentence-transformers embeddings
- **Question Answering**: AI-powered answers with source citations and confidence scores
- **Document Summarization**: Multi-document summarization with filtering options
- **Web Interface**: User-friendly Streamlit frontend
- **REST API**: FastAPI backend with comprehensive endpoints
- **Deployment Ready**: Docker support and deployment configurations

## 🏗️ Architecture

```
User Query → FastAPI → LangChain (Retriever → Reranker → Generator) → LLM → Response
```

### Component Breakdown

- **Ingestion**: `pdfplumber`, `pypdf` for document processing
- **Embeddings**: `sentence-transformers` (all-mpnet-base-v2)
- **Vector DB**: `FAISS` with HNSW indexing
- **Orchestration**: `LangChain` for pipeline management
- **Generation**: `google/flan-t5-base` for answer generation
- **Backend API**: `FastAPI` with CORS support
- **Frontend UI**: `Streamlit` with tabbed interface

## 📋 Requirements

- Python 3.11+
- 4GB+ RAM (8GB+ recommended)
- CPU-based inference (GPU optional)

## 🛠️ Installation

### Option 1: Local Setup

1. **Clone and setup**:
```bash
git clone <repository-url>
cd rag_app
pip install -r requirements.txt
```

2. **Start the backend**:
```bash
cd src/api
uvicorn app:app --host 0.0.0.0 --port 8001
```

3. **Start the frontend** (in another terminal):
```bash
cd demo
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Option 2: Docker

```bash
docker-compose up --build
```

Access the application at:
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## 🎯 Quick Start

1. **Upload Documents**: Use the "Upload Documents" tab to add PDF, TXT, or MD files
2. **Ask Questions**: Navigate to "Ask Questions" and enter your query
3. **Get Summaries**: Use "Summarize" to generate document overviews

### API Usage

```python
import requests

# Upload documents
files = {'files': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8001/ingest', files=files)

# Ask questions
query = {"question": "What is machine learning?", "max_results": 5}
response = requests.post('http://localhost:8001/query', json=query)
print(response.json()['answer'])

# Generate summary
summary_request = {"max_documents": 10}
response = requests.post('http://localhost:8001/summarize', json=summary_request)
print(response.json()['summary'])
```

## 📊 Performance Metrics

- **Retrieval Recall@5**: Measures document retrieval accuracy
- **Answer Relevance**: Quality of generated responses
- **Response Time**: End-to-end query latency (typically 2-5 seconds)
- **Confidence Scoring**: Automated confidence estimation for answers

## 🔧 Configuration

### Environment Variables

```bash
export TRANSFORMERS_CACHE=/path/to/cache
export PYTHONPATH=/path/to/src
```

### Model Configuration

Edit `src/generator.py` to change the generation model:
```python
generator = RAGGenerator(model_name="google/flan-t5-large")  # Larger model
```

Edit `src/embed.py` to change the embedding model:
```python
embedding_manager = EmbeddingManager(model_name='all-MiniLM-L6-v2')  # Faster model
```

## 📁 Project Structure

```
rag_app/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── ingest.py               # Document processing
│   ├── embed.py                # Embedding & FAISS indexing
│   ├── retriever.py            # LangChain retrieval
│   └── generator.py            # Text generation
├── demo/
│   └── streamlit_app.py        # Streamlit frontend
├── docs/
│   └── architecture.md         # Architecture documentation
├── deploy/
│   ├── Dockerfile              # Container configuration
│   └── docker-compose.yml      # Multi-service deployment
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── model_card.md              # Model documentation
├── dataset_card.md            # Data processing documentation
└── README.md                  # This file
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Manual Testing
```bash
# Test document ingestion
python src/ingest.py

# Test embedding generation
python src/embed.py

# Test retrieval
python src/retriever.py

# Test generation
python src/generator.py
```

## 🚀 Deployment

### Local Development
```bash
python run_demo.py
```

### Production Deployment

1. **Docker**:
```bash
docker build -t rag-app .
docker run -p 8000:8000 -p 8501:8501 rag-app
```

2. **Cloud Deployment**:
   - AWS ECS/EKS
   - Google Cloud Run
   - Azure Container Instances

### Environment Setup

For production, ensure:
- Sufficient memory allocation (8GB+ recommended)
- Persistent storage for vector indices
- Load balancing for multiple instances
- Monitoring and logging setup

## 📈 Monitoring

### Health Checks
- `GET /health` - System status and readiness
- Document processing metrics
- Query response times
- Error rates and logging

### Performance Optimization

- **Caching**: Recent query embeddings and results
- **Batching**: Process multiple documents together
- **Quantization**: Use smaller models for faster inference
- **Indexing**: Optimize FAISS parameters for your dataset

## 🔒 Security Considerations

- Documents are processed locally (no external API calls)
- Input validation for uploaded files
- Rate limiting on API endpoints
- CORS configuration for frontend access
- No persistent storage of sensitive data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project uses components with various licenses:
- **sentence-transformers**: Apache 2.0
- **FAISS**: MIT
- **Transformers**: Apache 2.0
- **LangChain**: MIT
- **FastAPI**: MIT
- **Streamlit**: Apache 2.0

## 🆘 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use smaller models
2. **Slow Performance**: Check CPU usage and consider GPU acceleration
3. **Import Errors**: Ensure all dependencies are installed
4. **Port Conflicts**: Change ports in configuration files

### Getting Help

- Check the [documentation](docs/)
- Review [model card](model_card.md) and [dataset card](dataset_card.md)
- Open an issue for bugs or feature requests

## 🎯 Roadmap

### Planned Features
- [ ] Support for additional file formats (DOCX, HTML)
- [ ] Multi-language support
- [ ] Advanced reranking models
- [ ] Real-time document updates
- [ ] User authentication and document management
- [ ] Advanced analytics and insights

### Performance Improvements
- [ ] GPU acceleration support
- [ ] Distributed processing
- [ ] Advanced caching strategies
- [ ] Model quantization options

---

**Built with ❤️ using FastAPI, LangChain, FAISS, and Streamlit**

For questions or support, please refer to the documentation or open an issue.


