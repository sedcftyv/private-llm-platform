#!/home/aida/vllm/private-llm-platform/.venv/bin/python
"""Validate the live OpenAI-compatible embedding endpoint."""

import argparse
import base64
import json
import math
import urllib.error
import urllib.request
from typing import Any


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="bge-small-zh-v1.5")
    args = parser.parse_args()

    try:
        float_response = _post(
            f"{args.base_url}/v1/embeddings",
            {
                "model": args.model,
                "input": ["住宿费超标如何处理？", "发现账号异常登录怎么办？"],
            },
        )
        base64_response = _post(
            f"{args.base_url}/v1/embeddings",
            {
                "model": args.model,
                "input": "住宿费超标如何处理？",
                "encoding_format": "base64",
            },
        )

        vectors = [item["embedding"] for item in float_response["data"]]
        dimensions = {len(vector) for vector in vectors}
        norms = [
            math.sqrt(sum(value * value for value in vector))
            for vector in vectors
        ]
        encoded = base64_response["data"][0]["embedding"]
        base64_dimensions = len(base64.b64decode(encoded)) // 4

        if float_response["model"] != args.model or dimensions != {512}:
            raise RuntimeError("unexpected embedding model or dimensions")
        if any(abs(norm - 1.0) > 1e-4 for norm in norms):
            raise RuntimeError("embedding vectors are not normalized")
        if base64_dimensions != 512:
            raise RuntimeError("base64 embedding has the wrong dimensions")
        if float_response["usage"]["prompt_tokens"] <= 0:
            raise RuntimeError("embedding usage was not populated")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"Day 4 embedding smoke test failed: HTTP {exc.code}: {body}")
        return 1
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Day 4 embedding smoke test failed: {exc}")
        return 1

    print("Day 4 embedding smoke test passed")
    print(f"model: {args.model}")
    print("batch size: 2")
    print("dimensions: 512")
    print("float and base64 formats: passed")
    print(f"prompt tokens: {float_response['usage']['prompt_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
