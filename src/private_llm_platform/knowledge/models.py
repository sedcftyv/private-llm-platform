"""Knowledge-base data models."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One independently retrievable document fragment."""

    chunk_id: str
    document_id: str
    document_title: str
    section: str
    content: str
    source_path: str
    version: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "DocumentChunk":
        """Restore a chunk from persisted metadata."""

        if not isinstance(value, dict):
            raise ValueError("chunk metadata entries must be objects")
        fields = (
            "chunk_id",
            "document_id",
            "document_title",
            "section",
            "content",
            "source_path",
            "version",
        )
        if any(not isinstance(value.get(field), str) for field in fields):
            raise ValueError("chunk metadata contains missing or invalid fields")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One vector-search result with its observable retrieval score."""

    chunk: DocumentChunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            **self.chunk.to_dict(),
            "retrieval_score": self.score,
            "rank": self.rank,
        }
