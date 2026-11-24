#!/home/aida/vllm/private-llm-platform/.venv/bin/python
"""Search a persisted FAISS knowledge base."""

import argparse
import json
from pathlib import Path

from private_llm_platform.knowledge.embedding import (
    SentenceTransformerEmbeddingBackend,
)
from private_llm_platform.knowledge.store import FaissKnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=Path("data/knowledge_base"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--model-name", default="bge-small-zh-v1.5")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    backend = SentenceTransformerEmbeddingBackend(
        args.model,
        model_name=args.model_name,
        device=args.device,
    )
    knowledge_base = FaissKnowledgeBase.load(args.index, backend)
    results = knowledge_base.search(args.query, limit=args.limit)
    print(
        json.dumps(
            [result.to_dict() for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
