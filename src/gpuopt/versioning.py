from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .config import get_settings
from .dependencies import (
    get_digital_twin,
    get_repository,
    get_scheduler_service,
    get_state_service,
)
from .domains.collectors import DomainCollector
from .domains.stores import RingStore, get_domain_store
from .healing.auto_healer import AutoHealer
from .model_governance.governance import ModelGovernor, get_governor
from .model_governance.models import ModelActionClass, ModelStatus, ModelVersion
from .optimizer.models import OptimizationRequest
from .optimizer.optimizer import Optimizer
from .policy.evolution import PolicyEvolutionEngine
from .predictor.failure_predictor import FailurePredictor
from .registry import get_registry
from .scheduler.rl_scheduler import Job, Node, RLScheduler
from .schemas import WorkloadRequirements

logger = logging.getLogger(__name__)


class JobSubmissionRequest(BaseModel):
    cluster_id: UUID
    job_name: str = Field(min_length=1, max_length=200)
    required_gpus: int = Field(default=1, ge=1, le=256)
    required_cpu_cores: Optional[int] = Field(default=1, ge=1)
    required_memory_gb: float = Field(default=8.0, ge=0.1)
    estimated_runtime_hours: float = Field(default=1.0, ge=0.1)
    priority: int = Field(default=5, ge=1, le=10)
    model_name: Optional[str] = None
    checkpointable: bool = False


class JobSubmissionResponse(BaseModel):
    status: str
    job_id: Optional[str] = None
    message: str
    simulation_results: Optional[dict] = None
    placement: Optional[dict] = None

V1_PREFIX = "/api/v1"
V2_PREFIX = "/api/v2"


def create_v2_router() -> APIRouter:
    router = APIRouter(prefix=V2_PREFIX, tags=["v2"])

    reg = get_registry()
    _rl: RLScheduler = reg.get_or_create("scheduler_v2", RLScheduler)
    _healer: AutoHealer = reg.get_or_create("healer_v2", AutoHealer)
    _predictor: FailurePredictor = reg.get_or_create("predictor_v2", FailurePredictor)
    _policy: PolicyEvolutionEngine = reg.get_or_create("policy_engine_v2", PolicyEvolutionEngine)
    _policy.initialize_population()

    def _gov() -> ModelGovernor:
        return get_governor()

    # ── System ──────────────────────────────────────────────

    @router.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @router.get("/version")
    def version() -> dict:
        return {"version": __version__, "api_version": "2"}

    # ── Clusters ────────────────────────────────────────────

    def _cluster_status(repo, cluster_id):
        from datetime import datetime, timezone
        state = repo.latest_state(cluster_id)
        if state is None:
            return "unknown"
        age = (datetime.now(timezone.utc) - state.collected_at).total_seconds()
        return "healthy" if age < 300 else "warning"

    def _cluster_dict(c, repo):
        return {
            "id": str(c.id),
            "name": c.name,
            "environment": c.environment,
            "connector_type": c.connector_type.value if hasattr(c.connector_type, "value") else str(c.connector_type),
            "description": c.description or "",
            "region": c.region or "",
            "status": _cluster_status(repo, c.id),
            "created_at": c.created_at.isoformat() if c.created_at else "",
        }

    @router.get("/clusters")
    def list_clusters() -> list[dict]:
        repo = get_repository()
        return [_cluster_dict(c, repo) for c in repo.list_clusters()]

    @router.get("/clusters/{cluster_id}")
    def get_cluster(cluster_id: UUID) -> dict:
        repo = get_repository()
        c = repo.get_cluster(cluster_id)
        if c is None:
            return {"error": "not_found"}
        return _cluster_dict(c, repo)

    # ── RL Scheduler ────────────────────────────────────────

    @router.post("/schedule")
    def schedule_job(data: dict) -> dict:
        nodes_data = data.get("nodes", [])
        if not nodes_data:
            return {"status": "error", "reason": "No nodes provided"}
        job = Job(
            id=data.get("id", ""),
            required_gpus=data.get("required_gpus", 1),
            priority=data.get("priority", 5),
            estimated_duration=data.get("estimated_duration", 1.0),
            memory_gb=data.get("memory_gb", 8.0),
            checkpointable=data.get("checkpointable", False),
        )
        nodes = [Node(**n) for n in nodes_data]
        result = _rl.schedule(job, nodes)
        if result.node:
            return {
                "status": "scheduled",
                "node_id": result.node_id,
                "explanation": result.reasoning,
                "reward": result.reward,
                "q_value": result.q_value,
            }
        return {"status": "queued", "reason": result.reasoning}

    @router.post("/train-scheduler")
    def train_scheduler(episodes: int = 100) -> dict:
        _rl.train_from_history(episodes)
        metrics = _rl.metrics()
        return {"status": "training_complete", "episodes": episodes, "metrics": metrics}

    @router.get("/scheduler-metrics")
    def scheduler_metrics() -> dict:
        return _rl.metrics()

    # ── Domains ─────────────────────────────────────────────

    @router.post("/domains/collect")
    def collect_domain(data: dict) -> dict:
        cluster_id = data.get("cluster_id", "default")
        store = get_domain_store()
        collector = DomainCollector(store)
        gpu = collector.collect_gpu_telemetry(cluster_id, data.get("node", "node-0"), data.get("gpu_count", 4))
        fabric = collector.collect_fabric_storage_telemetry(cluster_id)
        scheduler = collector.collect_scheduler_state(cluster_id)
        return {
            "gpu_telemetry": gpu.model_dump(mode="json"),
            "fabric_telemetry": fabric.model_dump(mode="json"),
            "scheduler_state": scheduler.model_dump(mode="json"),
        }

    @router.get("/domains/{domain_type}")
    def query_domain(domain_type: str, limit: int = 10) -> list[dict]:
        store = get_domain_store()
        attr_map: dict[str, str] = {
            "gpu": "gpu_node",
            "fabric": "fabric_storage",
            "scheduler": "scheduler_states",
            "training": "training_steps",
            "inference": "inference_samples",
            "tenant": "tenant_quotas",
            "cost": "cost_allocations",
            "actions": "action_events",
        }
        attr = attr_map.get(domain_type)
        if attr is None or not hasattr(store, attr):
            return []
        ring: RingStore = getattr(store, attr)
        events = ring.list(limit=limit)
        return [e.model_dump(mode="json") for e in events]

    @router.get("/domains")
    def domain_counts() -> dict:
        store = get_domain_store()
        return {
            name: ring.count()
            for name, ring in vars(store).items()
            if isinstance(ring, RingStore)
        }

    # ── Optimizer ───────────────────────────────────────────

    @router.post("/optimize")
    def optimize(data: dict) -> dict:
        optimizer = Optimizer()
        req = OptimizationRequest(**data)
        result = optimizer.optimize(req)
        return result.model_dump(mode="json")

    # ── Governance ──────────────────────────────────────────

    @router.post("/governance/models/register")
    def register_model(
        model_name: str,
        version: str,
        action_class: ModelActionClass,
        owner: str = "",
    ) -> ModelVersion:
        return _gov().register_model(model_name, version, action_class, owner)

    @router.get("/governance/models")
    def list_models(
        action_class: ModelActionClass | None = None,
        status: ModelStatus | None = None,
    ) -> list[ModelVersion]:
        return _gov().registry.list(action_class=action_class, status=status, limit=100)

    # ── Predictor ───────────────────────────────────────────

    @router.post("/predict")
    def predict_failure(telemetry: dict) -> dict:
        return _predictor.predict_failure(telemetry)

    @router.post("/train-predictor")
    def train_predictor(telemetry_data: List[dict], labels: List[int]) -> dict:
        return _predictor.train(telemetry_data, labels)

    # ── Policy ──────────────────────────────────────────────

    @router.post("/policy/evolve")
    def evolve_policies(metrics: List[dict]) -> dict:
        best = _policy.evolve(metrics)
        return {
            "status": "evolution_complete",
            "generations": _policy.generations,
            "best_fitness": best.fitness_score,
            "policy_rego": best.to_rego(),
        }

    @router.get("/policy/best")
    def get_best_policy() -> dict:
        if _policy.best_chromosome is not None:
            return {
                "status": "success",
                "policy": _policy.get_best_policy(),
                "fitness": _policy.best_chromosome.fitness_score,
                "generation": _policy.best_chromosome.generation,
            }
        return {"status": "no_policy_evolved_yet"}

    # ── Healing ─────────────────────────────────────────────

    @router.post("/healing/check")
    def check_health(telemetry: dict) -> dict:
        return _healer.check_node_health(telemetry)

    @router.post("/healing/execute")
    def execute_remediation(telemetry: dict, node_id: str = "unknown") -> dict:
        result = _healer.execute_remediation(telemetry, node_id)
        return {
            "node_id": result.node_id,
            "action": result.action.value,
            "status": result.status,
            "message": result.message,
            "duration_seconds": result.duration_seconds,
        }

    @router.get("/healing/history")
    def healing_history(limit: int = 50) -> list[dict]:
        return _healer.get_history(limit)

    # ── ML-Powered Job Submission ─────────────────────────

    @router.post("/submit-job", response_model=JobSubmissionResponse)
    def submit_job(req: JobSubmissionRequest) -> JobSubmissionResponse:
        cluster_id = req.cluster_id

        repo = get_repository()
        cluster = repo.get_cluster(cluster_id)
        if cluster is None:
            raise HTTPException(status_code=404, detail=f"Cluster not found: {cluster_id}")

        state_svc = get_state_service()
        state = state_svc.get_latest_state(cluster_id)
        if state is None:
            raise HTTPException(status_code=400, detail="No cluster state available. Collect state first.")

        free_gpus = sum(
            1 for n in state.nodes for g in n.gpu_devices
            if g.memory_used_bytes == 0
        )
        if free_gpus < req.required_gpus:
            return JobSubmissionResponse(
                status="rejected",
                message=f"Insufficient GPUs: requested {req.required_gpus}, available {free_gpus}",
                simulation_results={"free_gpus": free_gpus, "total_gpus": state.gpu_count},
            )

        telemetry = state.telemetry
        node_predictions: dict[str, dict] = {}
        if telemetry:
            for nt in telemetry.nodes:
                avg_gpu_util = 0.0
                avg_mem_util = 0.0
                avg_temp = 0.0
                avg_power = 0.0
                ecc = 0
                xid = 0
                devs = nt.gpu_devices
                if devs:
                    avg_gpu_util = sum(d.utilization_gpu_percent for d in devs) / len(devs)
                    avg_mem_util = sum(d.utilization_memory_percent for d in devs) / len(devs)
                    avg_temp = sum(d.temperature_gpu_celsius for d in devs) / len(devs)
                    avg_power = sum(d.power_draw_watts for d in devs) / len(devs)
                    ecc = sum(d.ecc_errors_volatile + d.ecc_errors_aggregate for d in devs)
                    xid = 0
                payload = {
                    "gpu_utilization": avg_gpu_util,
                    "memory_utilization": avg_mem_util,
                    "temperature": avg_temp,
                    "power_usage": avg_power,
                    "clock_speed": devs[0].clock_sm_mhz if devs else 0,
                    "ecc_errors": ecc,
                    "retired_pages": 0,
                    "xid_errors": xid,
                    "utilization_variance": 0.1,
                    "temperature_variance": 0.1,
                    "available_gpus": sum(1 for d in devs if d.memory_used_bytes == 0) if devs else 0,
                    "total_gpus": len(devs),
                    "queue_length": 0,
                    "job_failures": 0,
                    "job_retries": 0,
                    "average_job_duration": req.estimated_runtime_hours * 3600,
                }
                node_predictions[nt.node_name] = _predictor.predict_failure(payload)

        twin_svc = get_digital_twin()
        try:
            twin_svc.sync_twin(cluster_id)
        except KeyError:
            raise HTTPException(status_code=400, detail="Failed to sync digital twin")

        try:
            sch_svc = get_scheduler_service()
            placement = sch_svc.suggest_placement(
                cluster_id,
                WorkloadRequirements(
                    gpu_count=req.required_gpus,
                    gpu_memory_bytes=int(req.required_memory_gb * (1024**3)),
                    cpu_millicores=req.required_cpu_cores * 1000,
                    memory_bytes=int(req.required_memory_gb * (1024**3)),
                ),
            )
        except KeyError as exc:
            return JobSubmissionResponse(
                status="simulation_failed",
                message=f"Placement simulation failed: {exc}",
                simulation_results={"error": str(exc)},
            )

        high_risk_nodes: list[str] = []
        predictions_summary: dict[str, dict] = {}
        for node_name, pred in node_predictions.items():
            prob = pred.get("probability", 0)
            predictions_summary[node_name] = {
                "failure_predicted": pred.get("failure_predicted", False),
                "probability": prob,
                "risk_factors": pred.get("risk_factors", []),
            }
            if prob > 0.7:
                high_risk_nodes.append(node_name)

        if high_risk_nodes:
            return JobSubmissionResponse(
                status="rejected",
                message=f"High failure risk on nodes: {high_risk_nodes}",
                simulation_results={
                    "high_risk_nodes": high_risk_nodes,
                    "predictions": predictions_summary,
                    "placement": placement.model_dump(mode="json"),
                },
            )

        candidate_node = placement.suggested_node
        candidate_node_obj = next(
            (n for n in state.nodes if n.name == candidate_node),
            None,
        )
        total_on_node = len(candidate_node_obj.gpu_devices) if candidate_node_obj else 1
        free_on_node = sum(
            1 for g in candidate_node_obj.gpu_devices if g.memory_used_bytes == 0
        ) if candidate_node_obj else 0

        rl_nodes = [
            Node(
                id=n.name,
                available_gpus=sum(1 for g in n.gpu_devices if g.memory_used_bytes == 0),
                total_gpus=len(n.gpu_devices),
                free_memory_gb=sum(
                    (g.memory_total_bytes - g.memory_used_bytes) / (1024**3) for g in n.gpu_devices
                ) / max(len(n.gpu_devices), 1),
                gpu_model=n.gpu_devices[0].model if n.gpu_devices else "unknown",
            )
            for n in state.nodes
        ]
        rl_job = Job(
            id=f"{req.job_name}-{int(time.time())}",
            required_gpus=req.required_gpus,
            priority=req.priority,
            estimated_duration=req.estimated_runtime_hours,
            memory_gb=req.required_memory_gb,
            checkpointable=req.checkpointable,
        )
        rl_result = _rl.schedule(rl_job, rl_nodes)

        schedule_success = rl_result.success

        if not schedule_success:
            return JobSubmissionResponse(
                status="simulation_failed",
                message=f"RL scheduler could not schedule job: {rl_result.reasoning}",
                simulation_results={
                    "predictions": predictions_summary,
                    "placement": placement.model_dump(mode="json"),
                    "rl_result": {
                        "node_id": rl_result.node_id,
                        "reasoning": rl_result.reasoning,
                        "reward": rl_result.reward,
                        "q_value": rl_result.q_value,
                    },
                },
            )

        job_id = rl_job.id
        return JobSubmissionResponse(
            status="submitted",
            job_id=job_id,
            message=(
                f"Job submitted successfully to {candidate_node} "
                f"(free: {free_on_node}/{total_on_node} GPUs, "
                f"confidence: {placement.confidence:.0%})"
            ),
            simulation_results={
                "predictions": predictions_summary,
                "placement": placement.model_dump(mode="json"),
                "rl_result": {
                    "node_id": rl_result.node_id,
                    "reward": rl_result.reward,
                    "q_value": rl_result.q_value,
                    "reasoning": rl_result.reasoning,
                },
                "free_gpus": free_gpus,
                "total_gpus": state.gpu_count,
            },
            placement={
                "suggested_node": placement.suggested_node,
                "alternative_nodes": placement.alternative_nodes,
                "confidence": placement.confidence,
                "score": placement.score,
                "reasoning": placement.reasoning,
            },
        )

    return router


def mount_v2(app: FastAPI, v2_router: APIRouter) -> None:
    app.include_router(v2_router)
    logger.info("v2 API mounted at %s with %d routes", V2_PREFIX, len(v2_router.routes))


def build_v1_deprecation_header() -> dict[str, str]:
    return {
        "X-API-Version": "2",
        "X-API-Deprecation": "v1 will be removed after 2027-01-01; migrate to /api/v2",
    }


class V1DeprecationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(V1_PREFIX):
            response.headers["X-API-Version"] = "2"
            response.headers["X-API-Deprecation"] = "v1 will be removed after 2027-01-01; migrate to /api/v2"
        return response
