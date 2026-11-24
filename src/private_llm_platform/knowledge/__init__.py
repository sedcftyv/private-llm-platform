"""Document ingestion, embedding, and vector retrieval."""

from .chunking import chunk_document
from .embedding import EmbeddingBackend, EmbeddingResult
from .models import DocumentChunk, SearchResult
from .store import FaissKnowledgeBase

__all__ = [
    "DocumentChunk",
    "EmbeddingBackend",
    "EmbeddingResult",
    "FaissKnowledgeBase",
    "SearchResult",
    "chunk_document",
]
