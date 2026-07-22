from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gpuopt.references import TechnicalBasisService
from gpuopt.references_schemas import (
    AcquisitionFunction,
    AllocationRequest,
    BibliographyResponse,
    DcgmMetricsResponse,
    HydroHpoResponse,
    JanusMoeResponse,
    K8sGpuScheduleResponse,
    LyraScheduleResponse,
    ReferenceInfo,
    SkipDecodeResponse,
)

router = APIRouter(prefix="/api/v1/references", tags=["technical_basis"])
_basis = TechnicalBasisService()


@router.get("/bibliography", response_model=BibliographyResponse)
def get_bibliography() -> BibliographyResponse:
    return BibliographyResponse(
        references=_basis.get_bibliography(),
        total=len(_basis.get_bibliography()),
    )


@router.get("/bibliography/{key}", response_model=ReferenceInfo | None)
def get_reference(key: str) -> ReferenceInfo | None:
    for ref in _basis.get_bibliography():
        if ref.citation_key == key:
            return ref
    raise HTTPException(status_code=404, detail=f"Reference '{key}' not found")


@router.get("/k8s-gpu-scheduling", response_model=K8sGpuScheduleResponse)
def k8s_scheduling(num_nodes: int = 4) -> K8sGpuScheduleResponse:
    return _basis.k8s.schedule(num_nodes)


@router.post("/k8s-gpu-scheduling/allocate", response_model=dict)
def k8s_allocate(req: AllocationRequest) -> dict:
    nodes = _basis.k8s.get_node_inventory(4)
    result = _basis.k8s.allocate(req, nodes)
    return result.model_dump(mode="json")


@router.get("/dcgm-exporter", response_model=DcgmMetricsResponse)
def dcgm_metrics(num_gpus: int = 8) -> DcgmMetricsResponse:
    return _basis.dcgm.query(num_gpus)


@router.post("/hydro-hpo/suggest", response_model=HydroHpoResponse)
def hydro_suggest(acquisition: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT) -> HydroHpoResponse:
    return _basis.hydro.suggest(acquisition)


@router.post("/hydro-hpo/trial", response_model=dict)
def hydro_add_trial(hp: dict, score: float, duration: float = 0.0) -> dict:
    trial = _basis.hydro.add_trial(hp, score, duration)
    return trial.model_dump(mode="json")


@router.get("/lyra-scheduling", response_model=LyraScheduleResponse)
def lyra_schedule() -> LyraScheduleResponse:
    return _basis.lyra.schedule()


@router.get("/janus-moe", response_model=JanusMoeResponse)
def janus_moe(total_gpus: int = 8) -> JanusMoeResponse:
    return _basis.janus.analyze(total_gpus)


@router.get("/skip-decode", response_model=SkipDecodeResponse)
def skip_decode(total_layers: int = 32) -> SkipDecodeResponse:
    return _basis.skip.optimize(total_layers)


@router.get("/health")
def health() -> dict:
    return _basis.health()
