from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .schemas import AuditLogEntry

logger = logging.getLogger(__name__)


class AuditStore:
    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: list[AuditLogEntry] = []
        self._max_entries = max_entries

    def add(self, entry: AuditLogEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def list(self, limit: int = 100, offset: int = 0) -> list[AuditLogEntry]:
        return self._entries[offset:offset + limit]

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


_audit_store = AuditStore()


def get_audit_store() -> AuditStore:
    return _audit_store


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        user_id: str = getattr(request.state, "user_id", "system")
        username: str = getattr(request.state, "username", "system")

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        severity = "info"
        if response.status_code >= 500:
            severity = "error"
        elif response.status_code >= 400:
            severity = "warn"

        entry = AuditLogEntry(
            user_id=user_id,
            action=f"{request.method} {request.url.path}",
            resource_type=request.url.path.split("/")[3] if len(request.url.path.split("/")) > 3 else "unknown",
            resource_id=request.path_params.get("cluster_id", request.path_params.get("job_id", request.path_params.get("rule_id", request.path_params.get("user_id", "")))),
            details={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "username": username,
            },
            ip_address=request.client.host if request.client else "",
            severity=severity,
        )
        _audit_store.add(entry)

        if severity == "error":
            logger.error("audit", extra={"entry": entry.model_dump(mode="json")})

        return response
