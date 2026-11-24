"""FastAPI gateway for generation and embedding backends."""

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from private_llm_platform.knowledge.embedding import (
    EmbeddingBackend,
    EmbeddingBackendError,
    SentenceTransformerEmbeddingBackend,
)

from .config import GatewaySettings
from .routing import ModelRegistry, ModelRoute


def _request_id(request: Request) -> str:
    return request.state.request_id


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    retryable: bool,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "retryable": retryable,
            }
        },
    )


def _response_headers(
    response: httpx.Response,
    route: ModelRoute,
) -> dict[str, str]:
    headers = {
        "x-upstream-status": str(response.status_code),
        "x-model-route": route.name,
        "x-model-backend": route.backend,
        "x-upstream-model": route.upstream_model,
        "x-thinking-enabled": str(route.enable_thinking).lower(),
    }
    content_type = response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type
    return headers


def create_app(
    settings: GatewaySettings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    model_registry: ModelRegistry | None = None,
    embedding_backend: EmbeddingBackend | None = None,
) -> FastAPI:
    """Create a gateway application.

    Args:
        settings: Gateway configuration. Environment variables are used when omitted.
        upstream_transport: Optional HTTPX transport used by tests.
        model_registry: Optional logical model registry used by tests.
        embedding_backend: Optional embedding backend used by tests.

    Returns:
        Configured FastAPI application.
    """

    resolved_settings = settings or GatewaySettings.from_env()
    resolved_registry = model_registry or (
        ModelRegistry.from_file(resolved_settings.model_registry_path)
        if resolved_settings.model_registry_path
        else ModelRegistry.default()
    )
    resolved_embedding_backend = embedding_backend or (
        SentenceTransformerEmbeddingBackend(
            resolved_settings.embedding_model_path,
            model_name=resolved_settings.embedding_model_name,
            device=resolved_settings.embedding_device,
        )
    )
    registry_created = int(time())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            connect=resolved_settings.connect_timeout_seconds,
            read=resolved_settings.read_timeout_seconds,
            write=resolved_settings.read_timeout_seconds,
            pool=resolved_settings.connect_timeout_seconds,
        )
        async with httpx.AsyncClient(
            base_url=resolved_settings.upstream_base_url,
            timeout=timeout,
            transport=upstream_transport,
        ) as client:
            app.state.upstream_client = client
            yield

    application = FastAPI(
        title="Private LLM Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next: Any) -> Response:
        request.state.request_id = request.headers.get("x-request-id") or str(uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready")
    async def ready(request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.upstream_client
        try:
            upstream_response = await client.get("/health")
        except httpx.TimeoutException:
            return _error_response(
                status_code=504,
                code="MODEL_TIMEOUT",
                message="Generation backend health check timed out",
                request_id=_request_id(request),
                retryable=True,
            )
        except httpx.RequestError:
            return _error_response(
                status_code=503,
                code="MODEL_UNAVAILABLE",
                message="Generation backend is unavailable",
                request_id=_request_id(request),
                retryable=True,
            )

        if upstream_response.status_code != 200:
            return _error_response(
                status_code=503,
                code="MODEL_NOT_READY",
                message="Generation backend is not ready",
                request_id=_request_id(request),
                retryable=True,
            )
        return JSONResponse({"status": "ready"})

    @application.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        model_names = sorted(
            {*resolved_registry.names, resolved_embedding_backend.model_name}
        )
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": registry_created,
                    "owned_by": "private-llm-platform",
                }
                for name in model_names
            ],
        }

    @application.post("/v1/embeddings")
    async def create_embeddings(request: Request) -> Response:
        request_id = _request_id(request)
        try:
            payload = await request.json()
        except ValueError:
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request body must be valid JSON",
                request_id=request_id,
                retryable=False,
            )

        if not isinstance(payload, dict):
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request body must be a JSON object",
                request_id=request_id,
                retryable=False,
            )

        model = payload.get("model")
        if not isinstance(model, str) or not model:
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request field 'model' must be a non-empty string",
                request_id=request_id,
                retryable=False,
            )
        if model != resolved_embedding_backend.model_name:
            return _error_response(
                status_code=404,
                code="MODEL_NOT_FOUND",
                message=f"The model {model!r} does not exist",
                request_id=request_id,
                retryable=False,
            )

        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            inputs = [raw_input]
        elif (
            isinstance(raw_input, list)
            and raw_input
            and len(raw_input) <= 2048
            and all(isinstance(item, str) for item in raw_input)
        ):
            inputs = raw_input
        else:
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message=(
                    "Request field 'input' must be a string "
                    "or non-empty string array"
                ),
                request_id=request_id,
                retryable=False,
            )

        if any(not text.strip() for text in inputs):
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Embedding inputs must not be empty",
                request_id=request_id,
                retryable=False,
            )

        encoding_format = payload.get("encoding_format", "float")
        if encoding_format not in {"float", "base64"}:
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="encoding_format must be 'float' or 'base64'",
                request_id=request_id,
                retryable=False,
            )

        if "user" in payload and not isinstance(payload["user"], str):
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request field 'user' must be a string",
                request_id=request_id,
                retryable=False,
            )

        dimensions = payload.get("dimensions")
        if dimensions is not None and (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request field 'dimensions' must be a positive integer",
                request_id=request_id,
                retryable=False,
            )

        try:
            result = await run_in_threadpool(
                resolved_embedding_backend.encode_documents,
                inputs,
            )
        except EmbeddingBackendError:
            return _error_response(
                status_code=503,
                code="EMBEDDING_UNAVAILABLE",
                message="Embedding backend is unavailable",
                request_id=request_id,
                retryable=True,
            )

        if result.vectors.ndim != 2 or result.vectors.shape[0] != len(inputs):
            return _error_response(
                status_code=503,
                code="EMBEDDING_UNAVAILABLE",
                message="Embedding backend returned an invalid vector batch",
                request_id=request_id,
                retryable=True,
            )

        actual_dimensions = result.vectors.shape[1]
        if dimensions is not None and dimensions != actual_dimensions:
            return _error_response(
                status_code=400,
                code="UNSUPPORTED_DIMENSIONS",
                message=f"This model only supports {actual_dimensions} dimensions",
                request_id=request_id,
                retryable=False,
            )

        data: list[dict[str, Any]] = []
        for index, vector in enumerate(result.vectors):
            embedding: list[float] | str
            if encoding_format == "base64":
                embedding = base64.b64encode(
                    vector.astype("<f4", copy=False).tobytes()
                ).decode("ascii")
            else:
                embedding = vector.tolist()
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding,
                }
            )

        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "model": resolved_embedding_backend.model_name,
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "total_tokens": result.prompt_tokens,
                },
            }
        )

    @application.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        request_id = _request_id(request)
        try:
            payload = await request.json()
        except ValueError:
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request body must be valid JSON",
                request_id=request_id,
                retryable=False,
            )

        if not isinstance(payload, dict):
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request body must be a JSON object",
                request_id=request_id,
                retryable=False,
            )

        requested_model = payload.get("model")
        if not isinstance(requested_model, str) or not requested_model:
            return _error_response(
                status_code=400,
                code="INVALID_REQUEST",
                message="Request field 'model' must be a non-empty string",
                request_id=request_id,
                retryable=False,
            )

        route = resolved_registry.get(requested_model)
        if route is None:
            return _error_response(
                status_code=404,
                code="MODEL_NOT_FOUND",
                message=f"The model {requested_model!r} does not exist",
                request_id=request_id,
                retryable=False,
            )

        upstream_payload = route.apply(payload)

        client: httpx.AsyncClient = request.app.state.upstream_client
        upstream_request = client.build_request(
            "POST",
            "/v1/chat/completions",
            json=upstream_payload,
            headers={"x-request-id": request_id},
            timeout=httpx.Timeout(
                route.timeout_seconds,
                connect=resolved_settings.connect_timeout_seconds,
            ),
        )

        try:
            upstream_response = await client.send(
                upstream_request,
                stream=bool(upstream_payload.get("stream")),
            )
        except httpx.TimeoutException:
            return _error_response(
                status_code=504,
                code="MODEL_TIMEOUT",
                message="Generation request timed out",
                request_id=request_id,
                retryable=True,
            )
        except httpx.RequestError:
            return _error_response(
                status_code=503,
                code="MODEL_UNAVAILABLE",
                message="Generation backend is unavailable",
                request_id=request_id,
                retryable=True,
            )

        headers = _response_headers(upstream_response, route)
        if not upstream_payload.get("stream"):
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code,
                headers=headers,
            )

        if upstream_response.is_error:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            return Response(
                content=body,
                status_code=upstream_response.status_code,
                headers=headers,
            )

        async def stream_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                await upstream_response.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream_response.status_code,
            headers=headers,
        )

    return application


app = create_app()
