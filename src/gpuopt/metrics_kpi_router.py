from __future__ import annotations

from fastapi import APIRouter

from gpuopt.metrics_kpi import MetricsKpiDashboardService
from gpuopt.metrics_kpi_schemas import (
    EconomicsDecision,
    EconomicsMetrics,
    GpuDecision,
    GpuMetricsResult,
    MetricsDashboard,
    PlacementDecision,
    PlacementMetrics,
    ReliabilityDecision,
    ReliabilityMetrics,
    RequestDecision,
    RequestMetrics,
    ThermalDecision,
    ThermalMetrics,
)

router = APIRouter(prefix="/api/v1/metrics-kpi", tags=["metrics_kpi"])
_dashboard = MetricsKpiDashboardService()


@router.get("/dashboard", response_model=MetricsDashboard)
def get_dashboard(gpu_count: int = 8, num_nodes: int = 2, hourly_cost: float = 1.5) -> MetricsDashboard:
    return _dashboard.build_dashboard(gpu_count, num_nodes, hourly_cost)


@router.get("/request", response_model=dict)
def get_request_metrics() -> dict:
    from gpuopt.metrics_kpi import RequestMetricsCollector
    req = RequestMetricsCollector.collect()
    dec = RequestMetricsCollector.decide(req)
    return {"metrics": req.model_dump(mode="json"), "decision": dec.model_dump(mode="json")}


@router.get("/gpu", response_model=dict)
def get_gpu_metrics(num_gpus: int = 8) -> dict:
    from gpuopt.metrics_kpi import GpuMetricsCollector
    gpu = GpuMetricsCollector.collect(num_gpus)
    dec = GpuMetricsCollector.decide(gpu)
    return {"metrics": gpu.model_dump(mode="json"), "decision": dec.model_dump(mode="json")}


@router.get("/reliability", response_model=dict)
def get_reliability_metrics() -> dict:
    from gpuopt.metrics_kpi import ReliabilityMetricsCollector
    rel = ReliabilityMetricsCollector.collect()
    dec = ReliabilityMetricsCollector.decide(rel)
    return {"metrics": rel.model_dump(mode="json"), "decision": dec.model_dump(mode="json")}


@router.get("/thermal", response_model=dict)
def get_thermal_metrics(num_gpus: int = 8) -> dict:
    from gpuopt.metrics_kpi import ThermalMetricsCollector
    therm = ThermalMetricsCollector.collect(num_gpus)
    dec = ThermalMetricsCollector.decide(therm, num_gpus)
    return {"metrics": therm.model_dump(mode="json"), "decision": dec.model_dump(mode="json")}


@router.get("/placement", response_model=dict)
def get_placement_metrics(num_nodes: int = 4) -> dict:
    from gpuopt.metrics_kpi import PlacementMetricsCollector
    place = PlacementMetricsCollector.collect(num_nodes)
    dec = PlacementMetricsCollector.decide(place)
    return {"metrics": place.model_dump(mode="json"), "decision": dec.model_dump(mode="json")}


@router.get("/economics", response_model=dict)
def get_economics_metrics(gpu_count: int = 32, hourly_cost: float = 1.5) -> dict:
    from gpuopt.metrics_kpi import EconomicsMetricsCollector
    econ = EconomicsMetricsCollector.collect(gpu_count, hourly_cost)
    dec = EconomicsMetricsCollector.decide(econ)
    return {"metrics": econ.model_dump(mode="json"), "decision": dec.model_dump(mode="json")}


@router.get("/health")
def health() -> dict:
    return _dashboard.health()
