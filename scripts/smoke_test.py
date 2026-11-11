#!/home/aida/vllm/.venv/bin/python
"""Run Day 1 health, model discovery, and generation checks."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> tuple[int, Any]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else None


def check_health(base_url: str) -> None:
    request = urllib.request.Request(f"{base_url}/health", method="GET")
    with urllib.request.urlopen(request, timeout=10.0) as response:
        if response.status != 200:
            raise RuntimeError(f"health check returned HTTP {response.status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-4b")
    args = parser.parse_args()

    try:
        check_health(args.base_url)

        status, models = request_json(f"{args.base_url}/v1/models")
        model_ids = [item["id"] for item in models.get("data", [])]
        if status != 200 or args.model not in model_ids:
            raise RuntimeError(
                f"expected model {args.model!r}, received {model_ids!r}"
            )

        status, completion = request_json(
            f"{args.base_url}/v1/chat/completions",
            payload={
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "用一句话说明什么是大语言模型推理服务。",
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = completion["choices"][0]["message"]["content"]
        if status != 200 or not content.strip():
            raise RuntimeError("chat completion returned empty content")

    except (
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        urllib.error.URLError,
    ) as exc:
        print(f"Day 1 smoke test failed: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Day 1 smoke test failed: {exc}", file=sys.stderr)
        return 1

    print("Day 1 smoke test passed")
    print(f"models: {model_ids}")
    print(f"response: {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
