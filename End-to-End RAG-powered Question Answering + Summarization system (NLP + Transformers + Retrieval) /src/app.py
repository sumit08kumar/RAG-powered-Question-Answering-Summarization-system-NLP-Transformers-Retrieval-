"""
FastAPI application for RAG system.
Provides endpoints for document ingestion and question answering.
"""

import os
import sys
import logging
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Resolve project paths
BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_DIR = BASE_DIR / "vector_store"

# Import our modules from project root
sys.path.append(str(BASE_DIR))

try:
    from ingest import ingest_documents
    from retriever import create_retriever_from_chunks
    from generator import create_generator
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    ingest_documents = None
    create_retriever_from_chunks = None
    create_generator = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for system components
retriever: Any = None
generator: Any = None
system_status = {
    "initialized": False,
    "documents_loaded": 0,
    "index_ready": False,
    "last_error": None,
}

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str
    max_results: int = 5
    use_compression: bool = False

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: Optional[str] = None

class SummarizeRequest(BaseModel):
    query: Optional[str] = None
    max_documents: int = 10

class SummarizeResponse(BaseModel):
    summary: str
    num_documents: int
    input_length: int

class StatusResponse(BaseModel):
    status: str
    initialized: bool
    documents_loaded: int
    index_ready: bool
    message: str
    last_error: Optional[str] = None


async def startup_init():
    """Initialize the system on startup."""
    global generator, system_status
    try:
        logger.info("Initializing RAG system...")
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)

        if create_generator is not None:
            generator = create_generator()
            logger.info("Generator initialized")
        else:
            logger.warning("create_generator not available. Check dependencies and imports.")

        system_status["initialized"] = True
        logger.info("RAG system startup completed")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        system_status["initialized"] = False
        system_status["last_error"] = str(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_init()
    yield
    # Add any shutdown/cleanup here if needed.

# Initialize FastAPI app
app = FastAPI(
    title="RAG Question-Answering System",
    description="A Retrieval-Augmented Generation system for document Q&A and summarization",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "RAG Question-Answering System",
        "version": "1.0.0",
        "status": "running" if system_status["initialized"] else "initializing"
    }

@app.get("/health", response_model=StatusResponse)
async def health_check():
    return StatusResponse(
        status="healthy" if system_status["initialized"] else "initializing",
        initialized=system_status["initialized"],
        documents_loaded=system_status["documents_loaded"],
        index_ready=system_status["index_ready"],
        message="System is ready" if system_status["index_ready"] else "Upload documents to get started",
        last_error=system_status.get("last_error"),
    )

@app.post("/ingest")
async def ingest_documents_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    async_mode: bool = False,  # default sync for dev so index_ready flips immediately
):
    """
    Ingest documents and build the vector index.
    """
    global retriever, system_status

    if not system_status["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")

    if ingest_documents is None or create_retriever_from_chunks is None:
        raise HTTPException(status_code=500, detail="Server missing required components. Check dependencies and imports.")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    supported_extensions = {'.pdf', '.txt', '.md'}
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {supported_extensions}"
            )

    try:
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix, dir=str(VECTOR_DIR))
            content = await file.read()
            tmp.write(content)
            tmp.close()
            temp_files.append(tmp.name)

        # Update status immediately so UI shows uploaded count
        system_status["documents_loaded"] = system_status.get("documents_loaded", 0) + len(files)
        system_status["index_ready"] = False
        system_status["last_error"] = None

        if async_mode:
            # Background processing
            background_tasks.add_task(process_documents, temp_files, [f.filename for f in files])
            return {
                "message": f"Started processing {len(files)} documents",
                "files": [f.filename for f in files],
                "status": "processing"
            }
        else:
            # Synchronous processing (dev mode)
            process_documents(temp_file_paths=temp_files, original_filenames=[f.filename for f in files])
            return {
                "message": f"Processed {len(files)} documents",
                "files": [f.filename for f in files],
                "status": "completed",
                "index_ready": system_status["index_ready"],
                "last_error": system_status.get("last_error"),
            }

    except Exception as e:
        logger.error(f"Error in document ingestion: {e}")
        system_status["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")

def process_documents(temp_file_paths: List[str], original_filenames: List[str]):
    """Process documents and build index."""
    global retriever, system_status
    try:
        logger.info(f"Processing {len(temp_file_paths)} documents...")
        chunks = ingest_documents(temp_file_paths)
        logger.info(f"Created {len(chunks)} chunks from documents")

        if not chunks:
            raise RuntimeError("No chunks produced from uploaded documents.")

        retriever = create_retriever_from_chunks(chunks, save_path=str(VECTOR_DIR))

        system_status["index_ready"] = True
        system_status["last_error"] = None
        logger.info("Vector index created successfully")
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        system_status["index_ready"] = False
        system_status["last_error"] = str(e)
    finally:
        for p in temp_file_paths:
            try:
                os.unlink(p)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {p}: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    global retriever, generator

    if not system_status["index_ready"]:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first using /ingest endpoint."
        )
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not initialized.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        docs = retriever.retrieve_documents(request.question, use_compression=request.use_compression)
        if not docs:
            return QueryResponse(answer="I couldn't find any relevant information to answer your question.",
                                 confidence=0.0, sources=[])
        docs = docs[:request.max_results]
        result = generator.answer_question(request.question, docs)
        return QueryResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            sources=result['sources'],
            context_used=result.get('context_used')
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    global retriever, generator

    if not system_status["index_ready"]:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first using /ingest endpoint."
        )
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not initialized.")

    try:
        docs = (retriever.retrieve_documents(request.query, use_compression=False)
                if request.query else
                retriever.retrieve_documents("summary of all content", use_compression=False))

        docs = docs[:request.max_documents]
        if not docs:
            return SummarizeResponse(summary="No documents found to summarize.", num_documents=0, input_length=0)

        result = generator.summarize_documents(docs)
        return SummarizeResponse(summary=result['summary'],
                                 num_documents=result['num_documents'],
                                 input_length=result['input_length'])
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.get("/documents")
async def list_documents():
    if not system_status["index_ready"]:
        return {"message": "No documents loaded", "documents": []}
    return {
        "message": f"{system_status['documents_loaded']} documents loaded",
        "documents_count": system_status['documents_loaded'],
        "index_ready": system_status['index_ready']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")