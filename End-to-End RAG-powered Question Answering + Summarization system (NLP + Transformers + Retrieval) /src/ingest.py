"""
Document ingestion and preprocessing module for RAG system.
Handles PDF, text, and other document formats.
"""

import logging
from typing import List, Dict, Any, Union
from pathlib import Path

# Optional PDF dependencies
try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # type: ignore

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # type: ignore

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles document ingestion and preprocessing."""

    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.md']

    def ingest_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Ingest a single document and return processed content.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing processed content and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_type': file_path.suffix.lower()
        }

        if file_path.suffix.lower() == '.pdf':
            content = self._extract_pdf_content(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            content = self._extract_text_content(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return {
            'content': content,
            'metadata': metadata
        }

    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF file."""
        last_error = None

        # Try pdfplumber first
        if pdfplumber is not None:
            try:
                with pdfplumber.open(file_path) as pdf:
                    text_content = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                    text = '\n\n'.join(text_content).strip()
                    if text:
                        return text
            except Exception as e:
                last_error = e
                logger.warning(f"pdfplumber failed for {file_path}, trying pypdf: {e}")

        # Fallback to pypdf
        if PdfReader is not None:
            try:
                reader = PdfReader(file_path)
                text_content = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                text = '\n\n'.join(text_content).strip()
                if text:
                    return text
            except Exception as e:
                last_error = e
                logger.error(f"pypdf failed to extract content from {file_path}: {e}")

        # If neither method worked
        if last_error:
            raise RuntimeError(f"Failed to extract PDF content from {file_path}: {last_error}")
        raise RuntimeError(
            f"Failed to extract PDF content from {file_path}: "
            f"missing dependencies (install pdfplumber or pypdf)"
        )

    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from text files."""
        # Try UTF-8
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            pass

        # Try latin-1
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except UnicodeDecodeError:
            pass

        # Fallback with errors ignored
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Normalize whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Collapse excessive whitespace
        text = ' '.join(text.split())
        # Remove common artifacts
        text = text.replace('\x00', '').replace('\ufffd', '')
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk (in characters)
            overlap: Number of characters to overlap between chunks

        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text:
            return []

        if len(text) <= chunk_size:
            return [{'content': text, 'chunk_id': 0, 'start_pos': 0, 'end_pos': len(text)}]

        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_id = 0
        n = len(text)

        while start < n:
            end = min(start + chunk_size, n)

            # Try to end at a sentence or paragraph boundary near the end
            if end < n:
                window_start = max(start, end - 200)
                window = text[window_start:end]
                best = -1
                # Prefer paragraph breaks
                best = window.rfind('\n\n')
                if best == -1:
                    # Next try sentence end
                    best = max(window.rfind('. '), window.rfind('! '), window.rfind('? '))
                if best != -1:
                    end = window_start + best + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'chunk_id': chunk_id,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1

            if end >= n:
                break

            # Move start with overlap
            start = max(end - overlap, start + 1)

        return chunks


def ingest_documents(document_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Ingest multiple documents and return processed chunks.

    Args:
        document_paths: List of paths to document files

    Returns:
        List of processed document chunks with metadata
    """
    ingester = DocumentIngester()
    all_chunks: List[Dict[str, Any]] = []

    for doc_path in document_paths:
        try:
            logger.info(f"Processing document: {doc_path}")
            doc_data = ingester.ingest_document(doc_path)

            # Clean the text
            cleaned_text = ingester.clean_text(doc_data['content'])
            if not cleaned_text:
                logger.warning(f"No text extracted from {doc_path}, skipping.")
                continue

            # Chunk the text
            chunks = ingester.chunk_text(cleaned_text)

            # Add document metadata to each chunk
            for chunk in chunks:
                chunk['document_metadata'] = doc_data['metadata']
                chunk['source_document'] = doc_data['metadata']['filename']

            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {doc_path}")

        except Exception as e:
            logger.error(f"Failed to process document {doc_path}: {e}")
            continue

    return all_chunks


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest documents and print chunk stats.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Files, directories, or glob patterns to ingest (e.g., docs/*.pdf notes.txt)",
    )
    parser.add_argument(
        "--dir",
        dest="scan_dir",
        help="Directory to scan recursively for .pdf/.txt/.md (e.g., vector_store)",
    )
    parser.add_argument(
        "--base",
        dest="base",
        default=str(Path(__file__).resolve().parents[1]),
        help="Base directory for resolving relative paths (default: project root)",
    )
    args = parser.parse_args()

    base = Path(args.base)
    candidates: List[Path] = []

    def add_path(p: Path):
        if p.is_dir():
            for ext in ("*.pdf", "*.txt", "*.md"):
                candidates.extend(p.rglob(ext))
        else:
            # Expand glob patterns, or add file if exists
            if any(ch in p.name for ch in ["*", "?", "["]):
                for match in p.parent.glob(p.name):
                    candidates.append(match)
            elif p.exists():
                candidates.append(p)

    if args.scan_dir:
        scan = Path(args.scan_dir)
        add_path(scan if scan.is_absolute() else base / scan)

    for inp in args.inputs:
        p = Path(inp)
        add_path(p if p.is_absolute() else base / p)

    if not candidates:
        print("No input files found. Example usage:")
        print("  python3 src/ingest.py --dir vector_store")
        print("  python3 src/ingest.py docs/*.pdf notes.txt")
        raise SystemExit(1)

    files = [str(p) for p in sorted(set(candidates))]
    chunks = ingest_documents(files)
    print(f"Processed {len(chunks)} chunks from {len(files)} documents")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(f"Content: {chunk['content'][:200]}...")
        print(f"Source: {chunk['source_document']}")