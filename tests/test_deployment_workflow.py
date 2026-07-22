from __future__ import annotations

from gpuopt.deployment_workflow import DeploymentWorkflowService
from gpuopt.deployment_workflow_schemas import (
    Step1Input,
    Step1Output,
    Step2Input,
    Step2Output,
    Step3Input,
    Step3Output,
    Step4Input,
    Step4Output,
    Step5Input,
    Step5Output,
    Step6Input,
    Step6Output,
    Step7Input,
    Step7Output,
    WorkflowState,
    WorkflowStatus,
)


# ── Fixture ───────────────────────────────────────────────────

def _run_through_steps(svc: DeploymentWorkflowService) -> str:
    wf, s1 = svc.step1(Step1Input(model_name="meta-llama/Llama-3.1-8B-Instruct"))
    wid = wf.workflow_id
    svc.step2(Step2Input(gpu_model="NVIDIA H100-SXM-80GB"), wid)
    svc.step3(Step3Input(max_context_length=4096, expected_concurrent_sequences=4), wid)
    svc.step4(Step4Input(tensor_parallelism=1, pipeline_parallelism=1, num_replicas=1), wid)
    svc.step5(Step5Input(endpoint_url="http://localhost:8000/v1/chat/completions", num_requests=10), wid)
    svc.step6(Step6Input(measured_throughput_tokens_per_sec=1500.0, target_throughput_tokens_per_sec=5000.0, max_tokens_per_replica=1500.0), wid)
    svc.step7(Step7Input(), wid)
    return wid


# ── Unit: Step 1 - Model Identity ─────────────────────────────

def test_step1_known_model():
    svc = DeploymentWorkflowService()
    wf, out = svc.step1(Step1Input(model_name="llama-8b"))
    assert out.parameter_count_b == 8.0
    assert out.estimated_weight_gb > 0
    assert wf.current_step == 1


def test_step1_custom_model():
    svc = DeploymentWorkflowService()
    wf, out = svc.step1(Step1Input(model_name="custom/net-4b", parameter_count_b=4.0, precision="int8"))
    assert out.parameter_count_b == 4.0
    assert out.estimated_weight_gb < 10.0  # int8 compact
    assert out.model_card_summary.startswith("custom/net-4b")


def test_step1_fp32_weight():
    svc = DeploymentWorkflowService()
    _, out = svc.step1(Step1Input(model_name="test", parameter_count_b=1.0, precision="fp32"))
    assert round(out.estimated_weight_gb, 1) == 3.7  # 1B * 4 bytes / (1024^3)


# ── Unit: Step 2 - Hardware Specification ─────────────────────

def test_step2_basic():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    out = svc.step2(Step2Input(gpu_model="NVIDIA H100-SXM-80GB"), wf.workflow_id)[1]
    assert out.total_gpus_available == 8
    assert out.gpus_per_node == 8
    assert out.max_tensor_parallelism == 8


def test_step2_multi_node():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    out = svc.step2(Step2Input(num_nodes_available=4), wf.workflow_id)[1]
    assert out.total_gpus_available == 32


def test_step2_pcie_limits():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    out = svc.step2(Step2Input(interconnect="pcie", num_nodes_available=2), wf.workflow_id)[1]
    assert out.max_tensor_parallelism == out.gpus_per_node


# ── Unit: Step 3 - SLO Requirements ───────────────────────────

def test_step3_fits():
    svc = DeploymentWorkflowService()
    wid = _run_through_steps(svc)
    wf = svc.get_workflow(wid)
    assert wf is not None
    s3 = wf.step3_output
    assert s3 is not None
    assert s3.estimated_kv_cache_gb > 0
    assert s3.estimated_total_memory_gb > 0


def test_step3_large_context():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input(model_name="llama-8b"))
    _, out = svc.step3(Step3Input(max_context_length=131072, expected_concurrent_sequences=64), wf.workflow_id)
    assert out.estimated_kv_cache_gb > 10


# ── Unit: Step 4 - Deployment ─────────────────────────────────

def test_step4_manifest():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input(model_name="llama-8b"))
    svc.step2(Step2Input(), wf.workflow_id)
    svc.step3(Step3Input(), wf.workflow_id)
    _, out = svc.step4(Step4Input(tensor_parallelism=1, pipeline_parallelism=1, num_replicas=2), wf.workflow_id)
    assert out.total_gpus_required == 2
    assert "kind: Deployment" in out.manifest_yaml
    assert "replicas: 2" in out.manifest_yaml


def test_step4_with_hpa():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    svc.step2(Step2Input(), wf.workflow_id)
    svc.step3(Step3Input(), wf.workflow_id)
    _, out = svc.step4(Step4Input(enable_hpa=True), wf.workflow_id)
    assert "HorizontalPodAutoscaler" in out.manifest_yaml


# ── Unit: Step 5 - Benchmark ──────────────────────────────────

def test_step5_benchmark():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    svc.step2(Step2Input(), wf.workflow_id)
    svc.step3(Step3Input(), wf.workflow_id)
    svc.step4(Step4Input(), wf.workflow_id)
    _, out = svc.step5(Step5Input(endpoint_url="http://localhost:8000/v1/completions", num_requests=50), wf.workflow_id)
    assert out.benchmark.latency_p50_ms > 0
    assert out.benchmark.throughput_tokens_per_sec > 0
    assert out.endpoint_url == "http://localhost:8000/v1/completions"


def test_step5_dcgm():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    svc.step2(Step2Input(), wf.workflow_id)
    svc.step3(Step3Input(), wf.workflow_id)
    svc.step4(Step4Input(), wf.workflow_id)
    _, out = svc.step5(Step5Input(endpoint_url="http://localhost:8000", dcgm_metrics_available=True), wf.workflow_id)
    assert out.benchmark.dcgm_gpu_util_pct > 0


# ── Unit: Step 6 - Production Replica Count ───────────────────

def test_step6_replica_count():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    svc.step4(Step4Input(), wf.workflow_id)
    _, out = svc.step6(
        Step6Input(measured_throughput_tokens_per_sec=1000.0, target_throughput_tokens_per_sec=5000.0, max_tokens_per_replica=1000.0),
        wf.workflow_id,
    )
    assert out.required_replicas == 5
    assert out.recommended_replicas_with_buffer >= out.required_replicas


def test_step6_no_buffer():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    svc.step4(Step4Input(tensor_parallelism=1, pipeline_parallelism=1), wf.workflow_id)
    _, out = svc.step6(
        Step6Input(measured_throughput_tokens_per_sec=2000.0, target_throughput_tokens_per_sec=2000.0, max_tokens_per_replica=2000.0),
        wf.workflow_id,
    )
    assert out.required_replicas == 1
    assert out.total_gpus_required > 0


# ── Unit: Step 7 - Optimization Experiments ───────────────────

def test_step7_default_experiments():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input(model_name="llama-8b"))
    svc.step2(Step2Input(), wf.workflow_id)
    svc.step3(Step3Input(), wf.workflow_id)
    svc.step4(Step4Input(tensor_parallelism=1), wf.workflow_id)
    svc.step5(Step5Input(endpoint_url="http://localhost:8000"), wf.workflow_id)
    _, out = svc.step7(Step7Input(), wf.workflow_id)
    assert out.experiments_run == 3
    assert out.best_experiment is not None
    assert out.savings_pct > 0


def test_step7_custom_experiments():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    svc.step4(Step4Input(), wf.workflow_id)
    svc.step5(Step5Input(endpoint_url="http://localhost:8000"), wf.workflow_id)
    from gpuopt.deployment_workflow_schemas import OptimizationExperiment
    custom = [
        OptimizationExperiment(experiment_id="exp-custom-1", name="Custom A", description="", configuration={},
                               measured_throughput_tokens_per_sec=500.0, cost_per_million_tokens=0.5),
    ]
    _, out = svc.step7(Step7Input(experiments=custom), wf.workflow_id)
    assert out.experiments_run == 1
    assert out.best_experiment is not None
    assert out.best_experiment.experiment_id == "exp-custom-1"


# ── Workflow Lifecycle ────────────────────────────────────────

def test_full_workflow():
    svc = DeploymentWorkflowService()
    wid = _run_through_steps(svc)
    wf = svc.get_workflow(wid)
    assert wf is not None
    assert wf.status == WorkflowStatus.COMPLETED
    assert wf.step1_output is not None
    assert wf.step2_output is not None
    assert wf.step3_output is not None
    assert wf.step4_output is not None
    assert wf.step5_output is not None
    assert wf.step6_output is not None
    assert wf.step7_output is not None
    assert wf.current_step == 7


def test_workflow_list():
    svc = DeploymentWorkflowService()
    _run_through_steps(svc)
    _run_through_steps(svc)
    wfs = svc.list_workflows()
    assert len(wfs) >= 2


def test_workflow_delete():
    svc = DeploymentWorkflowService()
    wid = _run_through_steps(svc)
    assert svc.delete_workflow(wid) is True
    assert svc.get_workflow(wid) is None


def test_next_step_progression():
    svc = DeploymentWorkflowService()
    wf, _ = svc.step1(Step1Input())
    ns = svc.get_next_step(wf.workflow_id)
    assert ns["step"] == 2

    svc.step2(Step2Input(), wf.workflow_id)
    ns = svc.get_next_step(wf.workflow_id)
    assert ns["step"] == 3


def test_health():
    svc = DeploymentWorkflowService()
    h = svc.health()
    assert h["status"] == "healthy"
    assert h["workflows_active"] >= 0


# ── API Tests ─────────────────────────────────────────────────

def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/deployment-workflow/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


def test_step1_api_creates_workflow():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/deployment-workflow/step1", json={"model_name": "llama-8b"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert "workflow_id" in data
        assert data["current_step"] == 1
        assert float(data["output"]["parameter_count_b"]) == 8.0


def test_full_api_workflow():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r1 = client.post("/api/v1/deployment-workflow/step1", json={"model_name": "meta-llama/Llama-3.1-8B-Instruct"})
        wid = r1.json()["workflow_id"]

        r2 = client.post(f"/api/v1/deployment-workflow/{wid}/step2", json={})
        assert r2.status_code == 200

        r3 = client.post(f"/api/v1/deployment-workflow/{wid}/step3", json={"max_context_length": 4096})
        assert r3.status_code == 200

        r4 = client.post(f"/api/v1/deployment-workflow/{wid}/step4", json={"tensor_parallelism": 1})
        assert r4.status_code == 200
        assert "kind: Deployment" in r4.json()["output"]["manifest_yaml"]

        r5 = client.post(f"/api/v1/deployment-workflow/{wid}/step5", json={"endpoint_url": "http://localhost:8000/v1"})
        assert r5.status_code == 200

        r6 = client.post(f"/api/v1/deployment-workflow/{wid}/step6", json={
            "measured_throughput_tokens_per_sec": 1500.0,
            "target_throughput_tokens_per_sec": 5000.0,
            "max_tokens_per_replica": 1500.0,
        })
        assert r6.status_code == 200

        r7 = client.post(f"/api/v1/deployment-workflow/{wid}/step7", json={})
        assert r7.status_code == 200
        assert r7.json()["current_step"] == 7

        rget = client.get(f"/api/v1/deployment-workflow/{wid}")
        assert rget.status_code == 200
        assert rget.json()["status"] == "completed"


def test_get_missing_workflow():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/deployment-workflow/nonexistent-id")
        assert r.status_code == 404


def test_list_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/deployment-workflow/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


def test_next_step_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r1 = client.post("/api/v1/deployment-workflow/step1", json={})
        wid = r1.json()["workflow_id"]
        r = client.get(f"/api/v1/deployment-workflow/{wid}/next-step")
        assert r.status_code == 200
        assert r.json()["step"] == 2
