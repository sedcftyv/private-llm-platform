"""Tests for the HTTP gateway and logical model routes."""

import base64
import json
from collections.abc import AsyncIterator, Sequence

import httpx
import numpy as np
from fastapi.testclient import TestClient

from private_llm_platform.gateway.app import create_app
from private_llm_platform.gateway.config import GatewaySettings
from private_llm_platform.knowledge.embedding import EmbeddingResult


class MockAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class FakeEmbeddingBackend:
    model_name = "bge-small-zh-v1.5"

    def encode_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors = np.asarray(
            [[float(index + 1), 0.5, -0.5] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )
        return EmbeddingResult(vectors=vectors, prompt_tokens=7 * len(texts))

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        return self.encode_documents(texts)


def make_client(
    handler: httpx.MockTransport,
    *,
    embedding_backend: FakeEmbeddingBackend | None = None,
) -> TestClient:
    app = create_app(
        GatewaySettings(upstream_base_url="http://upstream.test"),
        upstream_transport=handler,
        embedding_backend=embedding_backend,
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


def test_models_lists_registered_generation_and_embedding_models() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    with make_client(transport) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == [
        "bge-small-zh-v1.5",
        "qwen3-fast",
        "qwen3-reasoning",
    ]
    assert all(isinstance(model["created"], int) for model in response.json()["data"])


def test_embeddings_returns_openai_compatible_float_vectors() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    with make_client(
        transport,
        embedding_backend=FakeEmbeddingBackend(),
    ) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-small-zh-v1.5",
                "input": ["住宿费如何报销", "账号异常怎么办"],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.5, -0.5]},
            {"object": "embedding", "index": 1, "embedding": [2.0, 0.5, -0.5]},
        ],
        "model": "bge-small-zh-v1.5",
        "usage": {"prompt_tokens": 14, "total_tokens": 14},
    }


def test_embeddings_supports_base64_encoding() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    with make_client(
        transport,
        embedding_backend=FakeEmbeddingBackend(),
    ) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-small-zh-v1.5",
                "input": "住宿费如何报销",
                "encoding_format": "base64",
            },
        )

    encoded = response.json()["data"][0]["embedding"]
    vector = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
    assert response.status_code == 200
    assert vector.tolist() == [1.0, 0.5, -0.5]


def test_embeddings_rejects_token_ids_and_unknown_model() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    with make_client(
        transport,
        embedding_backend=FakeEmbeddingBackend(),
    ) as client:
        token_response = client.post(
            "/v1/embeddings",
            json={"model": "bge-small-zh-v1.5", "input": [1, 2, 3]},
        )
        model_response = client.post(
            "/v1/embeddings",
            json={"model": "unknown", "input": "hello"},
        )
        missing_model_response = client.post(
            "/v1/embeddings",
            json={"input": "hello"},
        )
        dimensions_response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-small-zh-v1.5",
                "input": "hello",
                "dimensions": 2,
            },
        )

    assert token_response.status_code == 400
    assert token_response.json()["error"]["code"] == "INVALID_REQUEST"
    assert model_response.status_code == 404
    assert model_response.json()["error"]["code"] == "MODEL_NOT_FOUND"
    assert missing_model_response.status_code == 400
    assert missing_model_response.json()["error"]["code"] == "INVALID_REQUEST"
    assert dimensions_response.status_code == 400
    assert dimensions_response.json()["error"]["code"] == "UNSUPPORTED_DIMENSIONS"


def test_non_streaming_chat_completion_is_forwarded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["x-request-id"] == "request-123"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3-4b"
        assert payload["chat_template_kwargs"] == {
            "custom_option": "preserved",
            "enable_thinking": False,
        }
        assert request.extensions["timeout"]["read"] == 60.0
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
                "model": "qwen3-fast",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {
                    "custom_option": "preserved",
                    "enable_thinking": True,
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert response.headers["x-request-id"] == "request-123"
    assert response.headers["x-upstream-status"] == "200"
    assert response.headers["x-model-route"] == "qwen3-fast"
    assert response.headers["x-model-backend"] == "qwen3"
    assert response.headers["x-upstream-model"] == "qwen3-4b"
    assert response.headers["x-thinking-enabled"] == "false"


def test_reasoning_route_enables_thinking() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3-4b"
        assert payload["chat_template_kwargs"]["enable_thinking"] is True
        assert request.extensions["timeout"]["read"] == 120.0
        return httpx.Response(200, json={"choices": []})

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-reasoning",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    assert response.status_code == 200
    assert response.headers["x-model-route"] == "qwen3-reasoning"
    assert response.headers["x-thinking-enabled"] == "true"


def test_unknown_model_is_rejected_without_calling_upstream() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called")

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "unknown-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"
    assert response.json()["error"]["retryable"] is False


def test_missing_model_is_rejected_without_calling_upstream() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called")

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_chat_completion_reports_unavailable_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(httpx.MockTransport(handler)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-fast",
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
        assert payload["model"] == "qwen3-4b"
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
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
            "model": "qwen3-fast",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-model-route"] == "qwen3-fast"
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
