from __future__ import annotations

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class GPUOptException(Exception):
    """Base exception for GPUOpt application."""

    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ClusterNotFoundError(GPUOptException):
    """Raised when a cluster is not found."""

    def __init__(self, cluster_id: str):
        super().__init__(
            message=f"Cluster not found: {cluster_id}",
            status_code=404,
            details={"cluster_id": cluster_id},
        )


class ClusterAlreadyExistsError(GPUOptException):
    """Raised when trying to create a cluster that already exists."""

    def __init__(self, name: str):
        super().__init__(
            message=f"Cluster name already exists: {name}",
            status_code=409,
            details={"name": name},
        )


class CheckExecutionError(GPUOptException):
    """Raised when check execution fails."""

    def __init__(self, cluster_name: str, reason: str):
        super().__init__(
            message=f"Check execution failed for cluster {cluster_name}: {reason}",
            status_code=500,
            details={"cluster_name": cluster_name, "reason": reason},
        )


class ConnectorError(GPUOptException):
    """Raised when connector configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=400,
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI app."""

    @app.exception_handler(GPUOptException)
    async def gpuopt_exception_handler(request: Request, exc: GPUOptException) -> JSONResponse:
        logger.warning(
            "gpuopt_exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "message": exc.message,
                "details": exc.details,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": {"type": type(exc).__name__},
            },
        )
