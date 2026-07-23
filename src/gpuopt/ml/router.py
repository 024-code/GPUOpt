from __future__ import annotations

from typing import Any

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


@router.post("/train-from-cluster")
def train_from_cluster(max_samples: int = Query(default=500, ge=50, le=5000),
                       n_synthetic: int = Query(default=1000, ge=0, le=10000)) -> dict:
    return _engine.train_on_cluster_data(max_samples, n_synthetic)


@router.get("/data-collection-status")
def data_collection_status() -> dict:
    collector = _engine.data_collector
    store = collector.domain_store
    status: dict[str, Any] = {"sources": {}}
    if store:
        status["sources"]["domain_gpu_node"] = store.gpu_node.count()
        status["sources"]["domain_scheduler_states"] = store.scheduler_states.count()
        status["sources"]["domain_scheduler_events"] = store.scheduler_events.count()
        status["sources"]["domain_training_steps"] = store.training_steps.count()
        status["sources"]["domain_inference_samples"] = store.inference_samples.count()
    return status


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


@router.get("/datasets")
def list_web_datasets() -> list[dict]:
    return _engine.list_web_datasets()


@router.post("/datasets/download")
def download_web_dataset(name: str = Query(description="Dataset name from /datasets"),
                         force: bool = False) -> dict:
    return _engine.download_web_dataset(name, force=force)


@router.post("/datasets/train")
def train_on_web_datasets(
    sources: str | None = Query(default=None, description="Comma-separated dataset names"),
    max_samples: int = Query(default=5000, ge=100, le=50000),
    blend_with_cluster: bool = Query(default=True),
    synthetic_factor: float = Query(default=0.5, ge=0.0, le=2.0),
) -> dict:
    source_list = [s.strip() for s in sources.split(",")] if sources else None
    return _engine.train_on_web_data(source_list, max_samples, blend_with_cluster, synthetic_factor)


@router.post("/simulate-enhanced")
def simulate_enhanced(
    gpu_model: str = Query(default="NVIDIA H100-SXM-80GB"),
    num_gpus: int = Query(default=8, ge=1, le=256),
    num_nodes: int = Query(default=1, ge=1, le=32),
    steps: int = Query(default=100, ge=10, le=1000),
    workload_type: str = Query(default="llm_training"),
) -> dict:
    return _engine.enhanced_simulate(gpu_model, num_gpus, num_nodes, steps, workload_type)


@router.post("/simulate-enhanced-failure")
def simulate_enhanced_failure(
    scenario: str = Query(default="thermal_runaway"),
    gpu_model: str = Query(default="NVIDIA H100-SXM-80GB"),
    num_gpus: int = Query(default=8, ge=1, le=64),
) -> dict:
    return _engine.enhanced_simulate_failure(scenario, gpu_model, num_gpus)


@router.post("/schedule")
def schedule_job(
    name: str = Query(default="", max_length=200),
    required_gpus: int = Query(default=1, ge=1, le=256),
    required_memory_gib: float = Query(default=8.0, ge=0.1, le=1024),
    estimated_runtime_hours: float = Query(default=1.0, ge=0.1, le=8760),
    priority: int = Query(default=5, ge=1, le=10),
    workload_type: str = Query(default="llm_inference"),
    policy: str | None = Query(default=None, description="round_robin, least_loaded, risk_aware, thermal_aware, power_efficient, hybrid"),
) -> dict:
    return _engine.schedule_job(name, required_gpus, required_memory_gib,
                                 estimated_runtime_hours, priority, workload_type, policy)


@router.get("/cluster-health")
def cluster_health() -> dict:
    return _engine.get_cluster_health()


@router.post("/closed-loop-train")
def closed_loop_train(
    cycles: int = Query(default=3, ge=1, le=20),
    steps_per_episode: int = Query(default=80, ge=10, le=500),
    retrain_every: int = Query(default=1, ge=1, le=10),
    gpu_model: str = Query(default="NVIDIA H100-SXM-80GB"),
    num_nodes: int = Query(default=1, ge=1, le=8),
) -> list[dict]:
    return _engine.closed_loop_train(cycles, steps_per_episode, retrain_every, gpu_model, num_nodes)


@router.post("/compare-policies")
def compare_policies(
    steps: int = Query(default=60, ge=10, le=300),
    num_nodes: int = Query(default=1, ge=1, le=8),
) -> list[dict]:
    return _engine.compare_policies(steps, num_nodes)


@router.post("/optimize-policies")
def optimize_policies(
    iterations: int = Query(default=10, ge=5, le=100),
    steps_per_eval: int = Query(default=50, ge=10, le=200),
    num_nodes: int = Query(default=1, ge=1, le=8),
) -> dict:
    return _engine.optimize_policies(iterations, steps_per_eval, num_nodes)


@router.post("/power-cap-analysis")
def power_cap_analysis() -> list[dict]:
    return _engine.power_cap_analysis()


@router.get("/drain-recommendations")
def drain_recommendations() -> list[dict]:
    return _engine.drain_recommendations()


@router.get("/gpu-catalog")
def list_gpu_catalog(
    vendor: str | None = Query(default=None),
    segment: str | None = Query(default=None),
    min_vram: float | None = Query(default=None, ge=1),
    capabilities: str | None = Query(default=None, description="Comma-separated: av1_encode,ray_tracing,tensor_cores,ecc,nvlink"),
) -> list[dict]:
    return _engine.list_gpu_catalog(vendor, segment, min_vram, capabilities)


@router.get("/gpu-catalog/stats")
def gpu_catalog_stats() -> dict:
    return _engine.get_gpu_catalog_stats()


@router.get("/gpu-catalog/lookup")
def lookup_gpu(name: str = Query(description="GPU model name to look up")) -> dict | None:
    return _engine.lookup_gpu(name)


@router.post("/schedule-with-capability")
def schedule_job_with_capability(
    name: str = Query(default="", max_length=200),
    required_gpus: int = Query(default=1, ge=1, le=256),
    required_memory_gib: float = Query(default=8.0, ge=0.1, le=1024),
    estimated_runtime_hours: float = Query(default=1.0, ge=0.1, le=8760),
    priority: int = Query(default=5, ge=1, le=10),
    workload_type: str = Query(default="llm_inference"),
    policy: str | None = Query(default=None),
    required_capabilities: str | None = Query(default=None, description="Comma-separated capability requirements"),
) -> dict:
    return _engine.schedule_job_with_capability(
        name, required_gpus, required_memory_gib,
        estimated_runtime_hours, priority, workload_type,
        policy, required_capabilities,
    )


@router.get("/cpu-catalog")
def list_cpu_catalog(
    vendor: str | None = Query(default=None),
    socket: str | None = Query(default=None),
    min_cores: int | None = Query(default=None, ge=1),
    has_igpu: bool | None = Query(default=None),
    min_igpu_tflops: float | None = Query(default=None, ge=0),
) -> list[dict]:
    return _engine.list_cpu_catalog(vendor, socket, min_cores, has_igpu, min_igpu_tflops)


@router.get("/cpu-catalog/stats")
def cpu_catalog_stats() -> dict:
    return _engine.get_cpu_catalog_stats()


@router.get("/cpu-catalog/lookup")
def lookup_cpu(name: str = Query(description="CPU model name to look up")) -> dict | None:
    return _engine.lookup_cpu(name)


@router.get("/hardware-detect")
def hardware_detect() -> dict:
    return _engine.detect_hardware()


@router.get("/memory-detect")
def memory_detect() -> dict:
    return _engine.detect_memory()
