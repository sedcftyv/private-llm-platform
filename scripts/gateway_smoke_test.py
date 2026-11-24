#!/home/aida/vllm/private-llm-platform/.venv/bin/python
"""Run Day 3 gateway model-routing checks against a live vLLM backend."""

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
) -> tuple[int, Any, dict[str, str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.status, json.loads(body), headers


def collect_stream(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> tuple[str, bool, dict[str, str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    saw_done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
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

    return "".join(chunks), saw_done, headers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3-fast")
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
        models_status, models = get_json(f"{args.base_url}/v1/models")
        completion_status, completion, completion_headers = post_json(
            f"{args.base_url}/v1/chat/completions",
            base_payload,
        )
        content = completion["choices"][0]["message"]["content"]
        stream_content, saw_done, stream_headers = collect_stream(
            f"{args.base_url}/v1/chat/completions",
            {**base_payload, "stream": True},
        )
        reasoning_status, reasoning, reasoning_headers = post_json(
            f"{args.base_url}/v1/chat/completions",
            {
                **base_payload,
                "model": "qwen3-reasoning",
                "messages": [
                    {
                        "role": "user",
                        "content": "17乘以19等于多少？请只回答数字。",
                    }
                ],
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        reasoning_message = reasoning["choices"][0]["message"]
        reasoning_finish_reason = reasoning["choices"][0].get("finish_reason")
        reasoning_output = (
            reasoning_message.get("content")
            or reasoning_message.get("reasoning_content")
            or ""
        )

        if health_status != 200 or health != {"status": "ok"}:
            raise RuntimeError(f"unexpected health response: {health!r}")
        if ready_status != 200 or ready != {"status": "ready"}:
            raise RuntimeError(f"unexpected readiness response: {ready!r}")
        model_names = {model["id"] for model in models["data"]}
        if models_status != 200 or model_names != {
            "qwen3-fast",
            "qwen3-reasoning",
        }:
            raise RuntimeError(f"unexpected model registry: {models!r}")
        if completion_status != 200 or not content.strip():
            raise RuntimeError("non-streaming completion was empty")
        if completion_headers.get("x-model-route") != args.model:
            raise RuntimeError("fast route metadata was not returned")
        if completion_headers.get("x-thinking-enabled") != "false":
            raise RuntimeError("fast route did not disable thinking")
        if not stream_content.strip() or not saw_done:
            raise RuntimeError("streaming completion was incomplete")
        if stream_headers.get("x-model-route") != args.model:
            raise RuntimeError("streaming route metadata was not returned")
        if reasoning_status != 200 or reasoning_finish_reason != "stop":
            raise RuntimeError("reasoning completion did not finish normally")
        if not reasoning_output.rstrip().endswith("323"):
            raise RuntimeError("reasoning completion returned the wrong answer")
        if reasoning_headers.get("x-model-route") != "qwen3-reasoning":
            raise RuntimeError("reasoning route metadata was not returned")
        if reasoning_headers.get("x-thinking-enabled") != "true":
            raise RuntimeError("reasoning route did not enable thinking")

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Day 3 smoke test failed: HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except (OSError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Day 3 smoke test failed: {exc}", file=sys.stderr)
        return 1

    print("Day 3 gateway routing smoke test passed")
    print(f"non-streaming response: {content}")
    print(f"streaming response: {stream_content}")
    print("reasoning response: thinking enabled; final answer 323")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
