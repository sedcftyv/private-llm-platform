"""Logical model registry and request routing."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Configuration for one client-visible model alias."""

    name: str
    backend: str
    upstream_model: str
    enable_thinking: bool
    timeout_seconds: float

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build the upstream payload for this route."""

        upstream_payload = dict(payload)
        template_kwargs = payload.get("chat_template_kwargs")
        if not isinstance(template_kwargs, dict):
            template_kwargs = {}

        upstream_payload["model"] = self.upstream_model
        upstream_payload["chat_template_kwargs"] = {
            **template_kwargs,
            "enable_thinking": self.enable_thinking,
        }
        return upstream_payload


class ModelRegistry:
    """Read-only collection of logical model routes."""

    def __init__(self, routes: Mapping[str, ModelRoute]) -> None:
        if not routes:
            raise ValueError("model registry must define at least one route")
        self._routes = MappingProxyType(dict(routes))

    @classmethod
    def default(cls) -> "ModelRegistry":
        """Load the registry bundled with the gateway package."""

        resource = files(__package__).joinpath("models.toml")
        return cls.from_bytes(resource.read_bytes())

    @classmethod
    def from_file(cls, path: Path) -> "ModelRegistry":
        """Load a registry from a TOML file."""

        return cls.from_bytes(path.read_bytes())

    @classmethod
    def from_bytes(cls, content: bytes) -> "ModelRegistry":
        """Parse and validate TOML registry data."""

        document = tomllib.loads(content.decode("utf-8"))
        model_configs = document.get("models")
        if not isinstance(model_configs, dict):
            raise ValueError("model registry must contain a [models] table")

        routes: dict[str, ModelRoute] = {}
        for name, config in model_configs.items():
            if not isinstance(name, str) or not name:
                raise ValueError("model route names must be non-empty strings")
            if not isinstance(config, dict):
                raise ValueError(f"model route {name!r} must be a table")
            routes[name] = _parse_route(name, config)
        return cls(routes)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered logical model names in stable order."""

        return tuple(sorted(self._routes))

    def get(self, name: str) -> ModelRoute | None:
        """Resolve a logical model name."""

        return self._routes.get(name)


def _required_string(config: dict[str, Any], field: str, route: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} for model route {route!r} must be a string")
    return value


def _parse_route(name: str, config: dict[str, Any]) -> ModelRoute:
    enable_thinking = config.get("enable_thinking")
    if not isinstance(enable_thinking, bool):
        raise ValueError(
            f"enable_thinking for model route {name!r} must be a boolean"
        )

    timeout_seconds = config.get("timeout_seconds")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or timeout_seconds <= 0
    ):
        raise ValueError(
            f"timeout_seconds for model route {name!r} must be greater than zero"
        )

    return ModelRoute(
        name=name,
        backend=_required_string(config, "backend", name),
        upstream_model=_required_string(config, "upstream_model", name),
        enable_thinking=enable_thinking,
        timeout_seconds=float(timeout_seconds),
    )
