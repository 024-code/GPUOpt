from __future__ import annotations

from fastapi import APIRouter, Query

from gpuopt.environment_checks import EnvironmentChecksService
from pydantic import BaseModel

from gpuopt.environment_checks_schemas import (
    EnvironmentCheckCatalog,
    EnvironmentCheckRun,
    EnvironmentType,
)

router = APIRouter(prefix="/api/v1/environment-checks", tags=["environment_checks"])
_service = EnvironmentChecksService()


class RunChecksRequest(BaseModel):
    environment: EnvironmentType = EnvironmentType.SANDBOX


@router.get("/health")
def health() -> dict:
    return _service.health()


@router.get("/catalog", response_model=list[EnvironmentCheckCatalog])
def get_catalog(environment: EnvironmentType | None = Query(default=None)) -> list[EnvironmentCheckCatalog]:
    return _service.get_catalog(environment)


@router.post("/run", response_model=EnvironmentCheckRun)
def run_checks(body: RunChecksRequest | None = None) -> EnvironmentCheckRun:
    env = body.environment if body else EnvironmentType.SANDBOX
    return _service.run_checks(env)
