from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List
from uuid import UUID

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .config import get_settings
from .dependencies import get_repository
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

logger = logging.getLogger(__name__)

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

    @router.get("/clusters")
    def list_clusters() -> list[dict]:
        repo = get_repository()
        return [
            {
                "id": str(c.id),
                "name": c.name,
                "connector_type": c.connector_type.value if hasattr(c.connector_type, "value") else str(c.connector_type),
                "status": c.status.value if hasattr(c.status, "value") else str(c.status),
                "region": c.region or "",
                "created_at": c.created_at.isoformat() if c.created_at else "",
            }
            for c in repo.list_clusters()
        ]

    @router.get("/clusters/{cluster_id}")
    def get_cluster(cluster_id: UUID) -> dict:
        repo = get_repository()
        c = repo.get_cluster(cluster_id)
        if c is None:
            return {"error": "not_found"}
        return {
            "id": str(c.id),
            "name": c.name,
            "connector_type": c.connector_type.value if hasattr(c.connector_type, "value") else str(c.connector_type),
            "status": c.status.value if hasattr(c.status, "value") else str(c.status),
            "region": c.region or "",
            "created_at": c.created_at.isoformat() if c.created_at else "",
        }

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
