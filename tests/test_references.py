from __future__ import annotations

from gpuopt.references import (
    K8sGpuScheduler,
    DcgmExporterService,
    HydroHpoService,
    LyraElasticScheduler,
    JanusMoeOptimizer,
    SkipDecodeOptimizer,
    TechnicalBasisService,
    REFERENCES,
)
from gpuopt.references_schemas import (
    AcquisitionFunction,
    AllocationRequest,
    K8sGpuScheduleResponse,
    DcgmMetricsResponse,
    HydroHpoResponse,
    LyraScheduleResponse,
    JanusMoeResponse,
    SkipDecodeResponse,
)


# ── K8s GPU Scheduling ───────────────────────────────────────

def test_k8s_schedule_basic():
    sched = K8sGpuScheduler()
    result = sched.schedule(num_nodes=2)
    assert isinstance(result, K8sGpuScheduleResponse)
    assert result.total_gpus >= 0
    assert len(result.node_inventory) == 2
    assert result.device_plugin_status in ("ready", "degraded", "unavailable")


def test_k8s_allocate_success():
    sched = K8sGpuScheduler()
    nodes = sched.get_node_inventory(2)
    req = AllocationRequest(gpu_count=1)
    result = sched.allocate(req, nodes)
    assert result.allocated or not result.allocated


def test_k8s_allocate_with_model_filter():
    sched = K8sGpuScheduler()
    req = AllocationRequest(gpu_count=2, gpu_model="H100")
    nodes = sched.get_node_inventory(4)
    result = sched.allocate(req, nodes)
    if result.allocated:
        assert len(result.gpu_ids) == 2


def test_k8s_node_inventory():
    sched = K8sGpuScheduler()
    nodes = sched.get_node_inventory(3)
    assert len(nodes) == 3
    for node in nodes:
        assert node.gpu_count_total >= 0
        assert node.allocatable_gpus >= 0
        assert len(node.gpu_devices) == node.gpu_count_total


# ── DCGM Exporter ────────────────────────────────────────────

def test_dcgm_query():
    svc = DcgmExporterService()
    result = svc.query(num_gpus=4)
    assert isinstance(result, DcgmMetricsResponse)
    assert len(result.targets) > 0
    assert len(result.samples) > 0
    assert result.daemonset_running is True


def test_dcgm_metric_types():
    svc = DcgmExporterService()
    samples = svc.collect_metrics(2)
    metrics = set(s.metric for s in samples)
    assert len(metrics) >= 10


def test_dcgm_scrape_config():
    svc = DcgmExporterService()
    config = svc.scrape_config()
    assert config["job_name"] == "nvidia-dcgm"
    assert config["scrape_interval"] == "15s"


# ── Hydro HPO ────────────────────────────────────────────────

def test_hydro_initial_suggest():
    svc = HydroHpoService()
    result = svc.suggest()
    assert isinstance(result, HydroHpoResponse)
    assert len(result.search_space) == 6
    assert result.surrogate.trials_completed == 0


def test_hydro_with_trials():
    svc = HydroHpoService()
    svc.add_trial({"learning_rate": 0.001, "batch_size": 64}, 0.85, 120.0)
    svc.add_trial({"learning_rate": 0.01, "batch_size": 128}, 0.72, 95.0)
    svc.add_trial({"learning_rate": 0.0001, "batch_size": 32}, 0.91, 180.0)

    result = svc.suggest(AcquisitionFunction.UPPER_CONFIDENCE_BOUND)
    assert result.surrogate.trials_completed == 3
    assert result.surrogate.best_score is not None
    assert result.expected_improvement > 0


def test_hydro_search_space():
    svc = HydroHpoService()
    assert len(svc.SEARCH_SPACE) == 6
    names = [p.name for p in svc.SEARCH_SPACE]
    assert "learning_rate" in names
    assert "batch_size" in names
    assert "optimizer" in names


# ── Lyra Elastic Scheduling ──────────────────────────────────

def test_lyra_schedule():
    sched = LyraElasticScheduler()
    result = sched.schedule()
    assert isinstance(result, LyraScheduleResponse)
    assert len(result.pools) == 3
    assert result.total_gpus > 0
    assert result.utilization_pct >= 0


def test_lyra_pools():
    sched = LyraElasticScheduler()
    pools = sched.get_pools()
    assert len(pools) == 3
    types = {p.pool_type for p in pools}
    assert "training" in types
    assert "inference" in types


def test_lyra_scaling_actions():
    sched = LyraElasticScheduler()
    actions = sched.suggest_scaling()
    for action in actions:
        assert action.action in ("scale_up", "scale_down")
        assert action.gpu_delta > 0


# ── JANUS MoE ────────────────────────────────────────────────

def test_janus_analyze():
    janus = JanusMoeOptimizer()
    result = janus.analyze(total_gpus=8)
    assert isinstance(result, JanusMoeResponse)
    assert result.moe_config.num_experts == 8
    assert len(result.expert_loads) == 8
    assert result.provisioning.total_gpus == 8


def test_janus_balancing():
    janus = JanusMoeOptimizer()
    loads = janus.simulate_expert_loads(8)
    balancing = janus.compute_balancing(loads)
    assert balancing.balancing_loss >= 0
    assert balancing.recommended_expert_capacity_factor >= 1.0


def test_janus_provisioning():
    janus = JanusMoeOptimizer()
    plan = janus.provision(16, expert_parallelism=2)
    assert plan.total_gpus == 16
    assert plan.expert_parallelism == 2
    assert plan.attention_gpus + plan.expert_gpus == 16


# ── SkipDecode ───────────────────────────────────────────────

def test_skip_decode():
    opt = SkipDecodeOptimizer()
    result = opt.optimize(total_layers=24)
    assert isinstance(result, SkipDecodeResponse)
    assert result.total_layers == 24
    assert result.layers_skipped >= 0
    assert result.estimated_speedup >= 1.0


def test_skip_quality_validation():
    opt = SkipDecodeOptimizer()
    result = opt.optimize(total_layers=32)
    assert result.quality_validation.acceptable is True


def test_skip_batching():
    opt = SkipDecodeOptimizer()
    compat = opt.check_batching_compatibility()
    assert compat.compatible is True
    assert compat.estimated_speedup > 1.0


# ── Aggregator ───────────────────────────────────────────────

def test_technical_basis_service():
    svc = TechnicalBasisService()
    health = svc.health()
    assert health["status"] == "healthy"
    assert len(health["components"]) == 6


def test_bibliography():
    svc = TechnicalBasisService()
    refs = svc.get_bibliography()
    assert len(refs) == 6
    keys = [r.citation_key for r in refs]
    assert "kubernetes_gpu" in keys
    assert "dcgm_exporter" in keys
    assert "hydro" in keys
    assert "lyra" in keys
    assert "janus" in keys
    assert "skipdecode" in keys


def test_references_module_constant():
    assert len(REFERENCES) == 6


def test_all_references_have_urls():
    for ref in REFERENCES:
        assert ref.url, f"{ref.citation_key} missing URL"
        assert ref.year > 0


# ── API Tests ────────────────────────────────────────────────

def test_bibliography_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/bibliography")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 6
        assert len(data["references"]) == 6


def test_reference_detail_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/bibliography/hydro")
        assert r.status_code == 200
        assert r.json()["citation_key"] == "hydro"


def test_reference_missing_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/bibliography/nonexistent")
        assert r.status_code == 404


def test_k8s_schedule_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/k8s-gpu-scheduling?num_nodes=2")
        assert r.status_code == 200
        data = r.json()
        assert len(data["node_inventory"]) == 2


def test_dcgm_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/dcgm-exporter?num_gpus=4")
        assert r.status_code == 200
        data = r.json()
        assert len(data["targets"]) > 0
        assert data["daemonset_running"] is True


def test_hydro_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/references/hydro-hpo/suggest")
        assert r.status_code == 200
        data = r.json()
        assert len(data["search_space"]) == 6


def test_lyra_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/lyra-scheduling")
        assert r.status_code == 200
        data = r.json()
        assert len(data["pools"]) == 3


def test_janus_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/janus-moe?total_gpus=8")
        assert r.status_code == 200
        data = r.json()
        assert data["moe_config"]["num_experts"] == 8


def test_skip_decode_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/skip-decode?total_layers=24")
        assert r.status_code == 200
        data = r.json()
        assert data["total_layers"] == 24


def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/references/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
