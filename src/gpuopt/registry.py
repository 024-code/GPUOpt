from __future__ import annotations

from typing import Any


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(self, name: str, service: Any, *, force: bool = False) -> None:
        if name in self._services and not force:
            raise KeyError(f"Service '{name}' already registered")
        self._services[name] = service

    def get(self, name: str, default: Any = None) -> Any:
        return self._services.get(name, default)

    def get_or_create(self, name: str, factory: type, *args: Any, **kwargs: Any) -> Any:
        existing = self._services.get(name)
        if existing is not None:
            return existing
        instance = factory(*args, **kwargs)
        self._services[name] = instance
        return instance

    def remove(self, name: str) -> None:
        self._services.pop(name, None)

    def list(self) -> dict[str, Any]:
        return dict(self._services)

    def clear(self) -> None:
        self._services.clear()


_registry = ServiceRegistry()


def get_registry() -> ServiceRegistry:
    return _registry


def reset_registry() -> None:
    _registry.clear()
