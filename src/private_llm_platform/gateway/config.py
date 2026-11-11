"""Gateway configuration."""

import os
from dataclasses import dataclass


def _positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """Runtime settings for the gateway."""

    upstream_base_url: str = "http://127.0.0.1:8000"
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        """Load settings from environment variables."""

        return cls(
            upstream_base_url=os.getenv(
                "LLM_UPSTREAM_BASE_URL",
                "http://127.0.0.1:8000",
            ).rstrip("/"),
            connect_timeout_seconds=_positive_float(
                "LLM_UPSTREAM_CONNECT_TIMEOUT_SECONDS",
                5.0,
            ),
            read_timeout_seconds=_positive_float(
                "LLM_UPSTREAM_READ_TIMEOUT_SECONDS",
                120.0,
            ),
        )
