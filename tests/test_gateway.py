"""Tests for the Day 2 HTTP gateway."""

import json
from collections.abc import AsyncIterator

import httpx
from fastapi.testclient import TestClient

from private_llm_platform.gateway.app import create_app
from private_llm_platform.gateway.config import GatewaySettings


class MockAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def make_client(handler: httpx.MockTransport) -> TestClient:
    app = create_app(
        GatewaySettings(upstream_base_url="http://upstream.test"),
        upstream_transport=handler,
    )
    return TestClient(app)


def test_health_does_not_require_upstream() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    with make_client(transport) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_ready_checks_upstream_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200)

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_reports_unavailable_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.get("/ready", headers={"x-request-id": "test-request"})

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "MODEL_UNAVAILABLE",
        "message": "Generation backend is unavailable",
        "request_id": "test-request",
        "retryable": True,
    }


def test_non_streaming_chat_completion_is_forwarded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["x-request-id"] == "request-123"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3-4b"
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}}
                ],
            },
        )

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-request-id": "request-123"},
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert response.headers["x-request-id"] == "request-123"
    assert response.headers["x-upstream-status"] == "200"


def test_chat_completion_reports_unavailable_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert response.json()["error"]["retryable"] is True


def test_streaming_chat_completion_preserves_sse_body() -> None:
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        return httpx.Response(
            200,
            stream=MockAsyncByteStream(
                [sse_body[:25], sse_body[25:70], sse_body[70:]]
            ),
            headers={"content-type": "text/event-stream"},
        )

    with make_client(httpx.MockTransport(handler)) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body == sse_body


def test_invalid_json_is_rejected_without_calling_upstream() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called")

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
