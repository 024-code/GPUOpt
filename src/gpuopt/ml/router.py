from __future__ import annotations

from fastapi import APIRouter, Query

from .engine import MLEngine

router = APIRouter(prefix="/api/v1/ml", tags=["ml_engine"])
_engine = MLEngine()


@router.get("/health")
def health() -> dict:
    return _engine.health()


@router.post("/predict")
def predict_failure(telemetry: dict) -> dict:
    return _engine.predict_failure(telemetry)


@router.post("/train")
def train_ensemble(telemetry_data: list[dict] | None = None,
                   labels: list[int] | None = None,
                   n_synthetic: int = 2000) -> dict:
    return _engine.train_ensemble(telemetry_data, labels, n_synthetic)


@router.post("/analyze-cluster")
def analyze_cluster(cluster_id: str = "default", node_count: int = Query(default=8, ge=1, le=64)) -> dict:
    return _engine.analyze_cluster(cluster_id, node_count)


@router.get("/model-info")
def model_info() -> dict:
    return _engine.get_model_info()


@router.get("/feature-importance")
def feature_importance() -> dict:
    return _engine.get_feature_importance()


@router.post("/simulate")
def simulate(num_gpus: int = Query(default=8, ge=1, le=64),
             gpu_model: str = Query(default="NVIDIA H100-SXM-80GB"),
             workload_type: str = Query(default="llm_inference"),
             duration_steps: int = Query(default=60, ge=1, le=500)) -> dict:
    return _engine.simulate(num_gpus, gpu_model, workload_type, duration_steps)


@router.post("/simulate-failure")
def simulate_failure(scenario: str = Query(default="thermal_runaway"),
                     num_gpus: int = Query(default=8, ge=1, le=64)) -> dict:
    return _engine.simulate_failure(scenario, num_gpus)


@router.get("/workload-profiles")
def workload_profiles() -> list[dict]:
    return _engine.list_profiles()


@router.post("/automl/random-search")
def automl_random_search(model_type: str = Query(default="random_forest"),
                         n_iter: int = Query(default=20, ge=5, le=100),
                         n_samples: int = Query(default=1000, ge=100, le=10000)) -> dict:
    return _engine.automl_random_search(model_type, n_iter, n_samples)


@router.post("/automl/compare-models")
def automl_compare_models(n_samples: int = Query(default=1000, ge=100, le=10000)) -> list[dict]:
    return _engine.automl_compare_models(n_samples)


@router.get("/registry")
def registry_list() -> dict:
    return _engine.registry_list()


@router.get("/registry/{name}")
def registry_get(name: str, version: str | None = Query(default=None)) -> dict | None:
    return _engine.registry_get(name, version)


@router.post("/registry/{name}/{version}/promote")
def registry_promote(name: str, version: str, stage: str = Query(default="production")) -> dict | None:
    return _engine.registry_promote(name, version, stage)
