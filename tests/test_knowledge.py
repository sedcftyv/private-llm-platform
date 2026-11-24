"""Tests for document chunking and FAISS retrieval."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from private_llm_platform.knowledge.chunking import chunk_document
from private_llm_platform.knowledge.embedding import EmbeddingResult
from private_llm_platform.knowledge.models import DocumentChunk
from private_llm_platform.knowledge.store import FaissKnowledgeBase


class KeywordEmbeddingBackend:
    model_name = "keyword-test"

    def encode_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors = []
        for text in texts:
            vectors.append(
                [
                    float(any(word in text for word in ("住宿", "酒店", "报销"))),
                    float(any(word in text for word in ("密码", "账号", "登录"))),
                    float(any(word in text for word in ("泄露", "安全事件", "报告"))),
                ]
            )
        return EmbeddingResult(
            vectors=np.asarray(vectors, dtype=np.float32),
            prompt_tokens=sum(len(text) for text in texts),
        )


def _chunk(chunk_id: str, section: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="policy",
        document_title="制度",
        section=section,
        content=content,
        source_path="documents/policy.md",
        version="2026-11",
    )


def test_markdown_chunking_preserves_metadata_and_stable_ids(tmp_path: Path) -> None:
    document = tmp_path / "travel_policy.md"
    document.write_text(
        """# 差旅报销制度

## 住宿费超标处理

住宿费超过标准时需要提交情况说明并取得审批。

## 报销时限

员工应在行程结束后十五个工作日内提交申请。
""",
        encoding="utf-8",
    )

    first = chunk_document(
        document,
        version="2026-11",
        max_chars=80,
        overlap_chars=10,
    )
    second = chunk_document(
        document,
        version="2026-11",
        max_chars=80,
        overlap_chars=10,
    )

    assert first == second
    assert [chunk.section for chunk in first] == ["住宿费超标处理", "报销时限"]
    assert all(chunk.document_id == "travel-policy" for chunk in first)
    assert all(chunk.document_title == "差旅报销制度" for chunk in first)
    assert all(chunk.version == "2026-11" for chunk in first)
    assert all(chunk.chunk_id.startswith("travel-policy-") for chunk in first)


def test_long_paragraph_chunks_respect_size_limit(tmp_path: Path) -> None:
    document = tmp_path / "long.txt"
    document.write_text("安全制度" * 50, encoding="utf-8")

    chunks = chunk_document(document, max_chars=40, overlap_chars=10)

    assert len(chunks) > 1
    assert all(0 < len(chunk.content) <= 40 for chunk in chunks)


def test_faiss_index_round_trip_and_semantic_search(tmp_path: Path) -> None:
    backend = KeywordEmbeddingBackend()
    chunks = [
        _chunk("travel-1", "住宿", "酒店住宿超标需要提交说明后报销。"),
        _chunk("security-1", "账号", "账号异常登录后应立即修改密码。"),
        _chunk("security-2", "事件", "发生数据泄露应立即报告安全事件。"),
    ]
    index_path = tmp_path / "index"

    knowledge_base = FaissKnowledgeBase.build(chunks, backend)
    knowledge_base.save(index_path)
    restored = FaissKnowledgeBase.load(index_path, backend)
    results = restored.search("酒店费用如何报销", limit=2)

    assert restored.size == 3
    assert results[0].chunk.chunk_id == "travel-1"
    assert results[0].rank == 1
    assert results[0].score > results[1].score
    assert (index_path / "index.faiss").is_file()
    assert (index_path / "metadata.json").is_file()
    assert (index_path / "manifest.json").is_file()
