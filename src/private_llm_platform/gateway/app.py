"""FastAPI gateway for an OpenAI-compatible vLLM backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import GatewaySettings


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


def _response_headers(response: httpx.Response) -> dict[str, str]:
    headers = {"x-upstream-status": str(response.status_code)}
    content_type = response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type
    return headers


def create_app(
    settings: GatewaySettings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
) -> FastAPI:
    """Create a gateway application.

    Args:
        settings: Gateway configuration. Environment variables are used when omitted.
        upstream_transport: Optional HTTPX transport used by tests.

    Returns:
        Configured FastAPI application.
    """

    resolved_settings = settings or GatewaySettings.from_env()

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

        client: httpx.AsyncClient = request.app.state.upstream_client
        upstream_request = client.build_request(
            "POST",
            "/v1/chat/completions",
            json=payload,
            headers={"x-request-id": request_id},
        )

        try:
            upstream_response = await client.send(
                upstream_request,
                stream=bool(payload.get("stream")),
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

        headers = _response_headers(upstream_response)
        if not payload.get("stream"):
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
