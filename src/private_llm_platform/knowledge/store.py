"""Persistent FAISS knowledge base."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .embedding import EmbeddingBackend
from .models import DocumentChunk, SearchResult

_INDEX_FILENAME = "index.faiss"
_METADATA_FILENAME = "metadata.json"
_MANIFEST_FILENAME = "manifest.json"


class FaissKnowledgeBase:
    """Cosine-similarity retrieval over normalized FAISS vectors."""

    def __init__(
        self,
        *,
        backend: EmbeddingBackend,
        index: faiss.Index,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunk metadata have different sizes")
        self._backend = backend
        self._index = index
        self._chunks = tuple(chunks)

    @classmethod
    def build(
        cls,
        chunks: Sequence[DocumentChunk],
        backend: EmbeddingBackend,
    ) -> "FaissKnowledgeBase":
        """Embed chunks and create an in-memory index."""

        if not chunks:
            raise ValueError("at least one document chunk is required")
        result = backend.encode_documents([chunk.content for chunk in chunks])
        vectors = np.ascontiguousarray(result.vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("embedding backend returned an invalid vector batch")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(backend=backend, index=index, chunks=chunks)

    @classmethod
    def load(
        cls,
        directory: Path,
        backend: EmbeddingBackend,
    ) -> "FaissKnowledgeBase":
        """Load an index and validate it against the configured backend."""

        manifest = _read_json(directory / _MANIFEST_FILENAME)
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
            raise ValueError("unsupported knowledge-base manifest")
        if manifest.get("embedding_model") != backend.model_name:
            raise ValueError("index embedding model does not match the backend")

        raw_metadata = _read_json(directory / _METADATA_FILENAME)
        if not isinstance(raw_metadata, list):
            raise ValueError("knowledge-base metadata must be a list")
        chunks = [DocumentChunk.from_dict(item) for item in raw_metadata]
        index = faiss.read_index(str(directory / _INDEX_FILENAME))
        if manifest.get("dimension") != index.d:
            raise ValueError("index dimension does not match its manifest")
        if manifest.get("chunk_count") != len(chunks):
            raise ValueError("chunk metadata does not match its manifest")
        return cls(backend=backend, index=index, chunks=chunks)

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""

        return len(self._chunks)

    def save(self, directory: Path) -> None:
        """Persist vectors, metadata, and compatibility information."""

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / _INDEX_FILENAME))
        _write_json(
            directory / _METADATA_FILENAME,
            [chunk.to_dict() for chunk in self._chunks],
        )
        _write_json(
            directory / _MANIFEST_FILENAME,
            {
                "format_version": 1,
                "embedding_model": self._backend.model_name,
                "dimension": self._index.d,
                "chunk_count": len(self._chunks),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def search(self, query: str, *, limit: int = 20) -> list[SearchResult]:
        """Return the nearest chunks and their cosine similarity scores."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        result = self._backend.encode_queries([query])
        vector = np.ascontiguousarray(result.vectors, dtype=np.float32)
        if vector.shape != (1, self._index.d):
            raise ValueError("query embedding dimension does not match the index")
        faiss.normalize_L2(vector)
        scores, indices = self._index.search(vector, min(limit, self.size))
        return [
            SearchResult(
                chunk=self._chunks[index],
                score=float(score),
                rank=rank,
            )
            for rank, (score, index) in enumerate(
                zip(scores[0], indices[0], strict=True),
                start=1,
            )
            if index >= 0
        ]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
