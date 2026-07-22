from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import get_settings

logger = logging.getLogger(__name__)

SKIP_RATE_LIMIT_PATHS = {"/health/live", "/health/ready", "/metrics"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def _cleanup_old_entries(self) -> None:
        now = time.time()
        if now - self._last_cleanup < 300:
            return
        cutoff = now - 3600
        for ip in list(self._requests.keys()):
            self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]
            if not self._requests[ip]:
                del self._requests[ip]
        self._last_cleanup = now

    def _is_rate_limited(self, client_ip: str) -> tuple[bool, int, int]:
        settings = get_settings()
        now = time.time()
        minute_cutoff = now - 60
        hour_cutoff = now - 3600

        self._requests[client_ip] = [t for t in self._requests[client_ip] if t > hour_cutoff]
        minute_requests = [t for t in self._requests[client_ip] if t > minute_cutoff]

        if len(minute_requests) >= settings.rate_limit_per_minute:
            oldest_in_window = min(minute_requests)
            retry_after = int(oldest_in_window + 60 - now) + 1
            return True, retry_after, 0

        hour_requests = [t for t in self._requests[client_ip] if t > hour_cutoff]
        if len(hour_requests) >= settings.rate_limit_per_hour:
            oldest_in_window = min(hour_requests)
            retry_after = int(oldest_in_window + 3600 - now) + 1
            return True, retry_after, 0

        remaining = settings.rate_limit_per_minute - len(minute_requests)
        return False, 0, remaining

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in SKIP_RATE_LIMIT_PATHS:
            return await call_next(request)

        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"
        self._cleanup_old_entries()

        is_limited, retry_after, remaining = self._is_rate_limited(client_ip)

        if is_limited:
            logger.warning(
                "rate_limit_exceeded",
                extra={"client_ip": client_ip, "path": request.url.path},
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(settings.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)

        return response
