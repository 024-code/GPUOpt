from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import get_settings
from .dependencies import get_rbac_manager
from .rbac import Permission, RBACManager

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {
    "/health/live", "/health/ready", "/health/detailed",
    "/metrics", "/docs", "/openapi.json", "/redoc",
    "/api/version",
    "/api/v2/health", "/api/v2/version",
}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        start_time = time.perf_counter()

        logger.info(
            "request_started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_host": request.client.host if request.client else None,
            },
        )

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "request_completed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        return response


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()

        request.state.user_id = "system"
        request.state.user = None
        request.state.username = "system"

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        rbac = get_rbac_manager()
        api_key = request.headers.get(settings.api_key_header)

        if api_key:
            user = rbac.authenticate(api_key)
            if user is not None:
                request.state.user_id = user.id
                request.state.user = user
                request.state.username = user.username
                return await call_next(request)

            from .auth import ExternalAuthService

            external = ExternalAuthService(settings)
            if external._enabled and "Bearer" not in api_key:
                api_key = f"Bearer {api_key}"
            result = external.authenticate(api_key)
            if result.authenticated:
                request.state.user_id = f"external:{result.username}"
                request.state.username = result.username
                request.state.external_auth = result
                return await call_next(request)

            return JSONResponse(
                status_code=403,
                content={"error": "Invalid API key or token"},
            )

        if settings.api_keyless_mode:
            return await call_next(request)

        if settings.oauth2_token_url or settings.oidc_issuer_url or settings.ldap_server_url:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                from .auth import ExternalAuthService

                external = ExternalAuthService(settings)
                result = external.authenticate(auth_header)
                if result.authenticated:
                    request.state.user_id = f"external:{result.username}"
                    request.state.username = result.username
                    request.state.external_auth = result
                    return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"error": "Missing API key", "header": settings.api_key_header},
        )
