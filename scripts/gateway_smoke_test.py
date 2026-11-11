#!/home/aida/vllm/private-llm-platform/.venv/bin/python
"""Run Day 2 gateway health, JSON, and streaming checks."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def get_json(url: str, timeout: float = 10.0) -> tuple[int, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def collect_stream(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> tuple[str, bool]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    saw_done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not response.headers.get_content_type() == "text/event-stream":
            raise RuntimeError("streaming response is not text/event-stream")

        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            delta = event["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                chunks.append(content)

    return "".join(chunks), saw_done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3-4b")
    args = parser.parse_args()

    base_payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "用一句话说明模型网关的作用。",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        health_status, health = get_json(f"{args.base_url}/health")
        ready_status, ready = get_json(f"{args.base_url}/ready")
        completion_status, completion = post_json(
            f"{args.base_url}/v1/chat/completions",
            base_payload,
        )
        content = completion["choices"][0]["message"]["content"]
        stream_content, saw_done = collect_stream(
            f"{args.base_url}/v1/chat/completions",
            {**base_payload, "stream": True},
        )

        if health_status != 200 or health != {"status": "ok"}:
            raise RuntimeError(f"unexpected health response: {health!r}")
        if ready_status != 200 or ready != {"status": "ready"}:
            raise RuntimeError(f"unexpected readiness response: {ready!r}")
        if completion_status != 200 or not content.strip():
            raise RuntimeError("non-streaming completion was empty")
        if not stream_content.strip() or not saw_done:
            raise RuntimeError("streaming completion was incomplete")

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Day 2 smoke test failed: HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except (OSError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Day 2 smoke test failed: {exc}", file=sys.stderr)
        return 1

    print("Day 2 gateway smoke test passed")
    print(f"non-streaming response: {content}")
    print(f"streaming response: {stream_content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
