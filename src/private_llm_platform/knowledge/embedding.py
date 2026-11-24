"""Embedding backend abstractions and Sentence Transformers implementation."""

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot serve a request."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Normalized vectors and their tokenizer usage."""

    vectors: NDArray[np.float32]
    prompt_tokens: int


class EmbeddingBackend(Protocol):
    """Backend contract shared by the API and retrieval pipeline."""

    @property
    def model_name(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingResult: ...


class SentenceTransformerEmbeddingBackend:
    """Lazily loaded Sentence Transformers backend intended for CPU serving."""

    def __init__(
        self,
        model_path: str,
        *,
        model_name: str,
        device: str = "cpu",
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
    ) -> None:
        self._model_path = model_path
        self._model_name = model_name
        self._device = device
        self._query_instruction = query_instruction
        self._model: Any = None
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        instructed = [f"{self._query_instruction}{text}" for text in texts]
        return self._encode(instructed)

    def _encode(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            raise ValueError("at least one input is required")

        with self._lock:
            model = self._load()
            try:
                vectors = model.encode(
                    list(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                tokenized = model.tokenizer(
                    list(texts),
                    padding=False,
                    truncation=True,
                )
            except Exception as exc:
                raise EmbeddingBackendError("embedding inference failed") from exc

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        prompt_tokens = sum(len(token_ids) for token_ids in tokenized["input_ids"])
        return EmbeddingResult(vectors=array, prompt_tokens=prompt_tokens)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_path,
                device=self._device,
            )
        except Exception as exc:
            raise EmbeddingBackendError("embedding model could not be loaded") from exc
        return self._model
