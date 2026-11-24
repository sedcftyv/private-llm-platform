#!/home/aida/vllm/private-llm-platform/.venv/bin/python
"""Build a persistent FAISS index from Markdown and text documents."""

import argparse
from pathlib import Path

from private_llm_platform.knowledge.chunking import chunk_document
from private_llm_platform.knowledge.embedding import (
    SentenceTransformerEmbeddingBackend,
)
from private_llm_platform.knowledge.store import FaissKnowledgeBase


def _document_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_dir():
            paths.update(
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in {".md", ".txt"}
            )
        elif input_path.suffix.lower() in {".md", ".txt"}:
            paths.add(input_path)
        else:
            raise ValueError(f"unsupported document path: {input_path}")
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/knowledge_base"))
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--model-name", default="bge-small-zh-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--version", default="2026-11")
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--overlap-chars", type=int, default=80)
    args = parser.parse_args()

    paths = _document_paths(args.inputs)
    if not paths:
        parser.error("no Markdown or text documents found")

    chunks = [
        chunk
        for path in paths
        for chunk in chunk_document(
            path,
            version=args.version,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    ]
    backend = SentenceTransformerEmbeddingBackend(
        args.model,
        model_name=args.model_name,
        device=args.device,
    )
    knowledge_base = FaissKnowledgeBase.build(chunks, backend)
    knowledge_base.save(args.output)

    print(f"indexed documents: {len(paths)}")
    print(f"indexed chunks: {knowledge_base.size}")
    print(f"embedding model: {backend.model_name}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
