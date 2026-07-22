from __future__ import annotations

from fastapi import APIRouter

from gpuopt.optimization_analysis import OptimizationAnalysisService
from gpuopt.optimization_analysis_schemas import OptimizationAnalysisResult, SyntheticObservation

router = APIRouter(prefix="/api/v1/optimization-analysis", tags=["optimization_analysis"])
_service = OptimizationAnalysisService()


@router.get("/health")
def health() -> dict:
    return _service.health()


@router.post("/analyze", response_model=OptimizationAnalysisResult)
def analyze(observation: SyntheticObservation | None = None) -> OptimizationAnalysisResult:
    return _service.analyze(observation)
