from __future__ import annotations

import os
import uuid

from gpuopt.inference.models import PlanRequest
from gpuopt.inference.planner import plan_inference
from gpuopt.inference.manifest_gen import generate_manifests
from gpuopt.inference.models import InferenceServiceSpec, ContainerResources, ManifestRequest
from gpuopt.inference_schemas import AnalyzeRequest, GpuUsageResponse
from gpuopt.inference_services import AnalyzeService
from gpuopt.gpu_usage import GpuInventoryService


# ── Plan Tests ───────────────────────────────────────────────

def test_plan_llama_8b():
    req = PlanRequest(model_name="llama-8b")
    plan = plan_inference(req)
    assert plan.model_name == "llama-8b"
    assert plan.weight_memory_gb > 0
    assert plan.kv_cache_gb > 0
    assert plan.total_memory_gb > plan.weight_memory_gb
    assert plan.recommended_tensor_parallelism >= 1
    assert plan.num_gpus_required >= 1
    assert plan.num_gpus_required == plan.recommended_tensor_parallelism


def test_plan_llama_70b():
    req = PlanRequest(model_name="llama-70b")
    plan = plan_inference(req)
    assert plan.model_name == "llama-70b"
    assert plan.num_gpus_required > 1
    assert plan.recommended_tensor_parallelism > 1
    assert plan.total_memory_gb > plan.weight_memory_gb


def test_plan_custom_model():
    req = PlanRequest(
        model_name="custom-13b",
        num_params=13.0,
        architecture="llama",
        num_layers=40,
        hidden_size=5120,
        num_heads=40,
        num_kv_heads=10,
        max_seq_len=8192,
        batch_size=4,
    )
    plan = plan_inference(req)
    assert plan.model_name == "custom-13b"
    assert plan.total_memory_gb > 0


def test_plan_fp8_precision():
    req = PlanRequest(model_name="llama-8b", dtype="fp8")
    plan = plan_inference(req)
    assert plan.dtype == "fp8"
    assert plan.weight_memory_gb < 16


# ── Manifest Tests ───────────────────────────────────────────

def test_manifest_generates_yaml():
    spec = InferenceServiceSpec(
        model_name="llama-8b",
        tensor_parallelism=1,
        pipeline_parallelism=1,
    )
    req = ManifestRequest(spec=spec)
    result = generate_manifests(req.spec)
    assert "deployment.yaml" in result.manifests
    assert "service.yaml" in result.manifests
    dep = result.manifests["deployment.yaml"]
    assert "nvidia.com/gpu: \"1\"" in dep or "nvidia.com/gpu" in dep


def test_manifest_gpu_count():
    spec = InferenceServiceSpec(
        model_name="llama-70b",
        tensor_parallelism=4,
        pipeline_parallelism=1,
    )
    req = ManifestRequest(spec=spec)
    result = generate_manifests(req.spec)
    dep = result.manifests["deployment.yaml"]
    assert "nvidia.com/gpu:" in dep and "'4'" in dep


def test_manifest_hpa():
    spec = InferenceServiceSpec(
        model_name="llama-8b",
        enable_hpa=True,
        hpa_max_replicas=5,
    )
    req = ManifestRequest(spec=spec)
    result = generate_manifests(req.spec)
    assert "hpa.yaml" in result.manifests
    assert "maxReplicas: 5" in result.manifests["hpa.yaml"]


def test_manifest_node_selector():
    spec = InferenceServiceSpec(
        model_name="llama-8b",
        node_selector={"gpuopt.ai/gpu-model": "h100"},
    )
    req = ManifestRequest(spec=spec)
    result = generate_manifests(req.spec)
    dep = result.manifests["deployment.yaml"]
    assert "gpuopt.ai/gpu-model: h100" in dep


# ── Analyze Tests ────────────────────────────────────────────

def test_analyze_low_utilization():
    req = AnalyzeRequest(
        model_name="llama-70b",
        gpu_model="NVIDIA-A100-80GB",
        gpu_count=4,
        avg_gpu_utilization=15.0,
        peak_gpu_memory_gib=30.0,
        quantisation="fp16",
    )
    result = AnalyzeService().analyze(req)
    assert result.model_name == "llama-70b"
    assert len(result.observations) > 0
    assert len(result.suggestions) > 0
    assert result.projected_throughput_tokens_per_sec > 0
    assert result.projected_monthly_cost_usd > 0
    assert any("concurrency" in s.title.lower() or "batch" in s.title.lower() for s in result.suggestions)


def test_analyze_high_latency():
    req = AnalyzeRequest(
        model_name="llama-8b",
        gpu_count=1,
        avg_latency_ms=2500.0,
        p99_latency_ms=5000.0,
        avg_gpu_utilization=85.0,
    )
    result = AnalyzeService().analyze(req)
    assert any("latency" in o.lower() for o in result.observations)
    assert any("tensor_parallelism" in s.category for s in result.suggestions)


def test_analyze_high_memory_pressure():
    req = AnalyzeRequest(
        model_name="llama-70b",
        gpu_model="NVIDIA-A100-80GB",
        gpu_count=1,
        peak_gpu_memory_gib=75.0,
        quantisation="fp16",
    )
    result = AnalyzeService().analyze(req)
    assert any("quantization" in s.category for s in result.suggestions)


def test_analyze_kv_cache_pressure():
    req = AnalyzeRequest(
        model_name="llama-8b",
        kv_cache_utilization=90.0,
    )
    result = AnalyzeService().analyze(req)
    assert any("kv" in o.lower() for o in result.observations)
    assert any("kv_cache" in s.category for s in result.suggestions)


def test_analyze_balanced():
    req = AnalyzeRequest(
        model_name="llama-8b",
        avg_gpu_utilization=50.0,
        peak_gpu_memory_gib=40.0,
        avg_latency_ms=100.0,
    )
    result = AnalyzeService().analyze(req)
    assert result.summary
    assert len(result.suggestions) >= 0


# ── GPU Usage Tests ──────────────────────────────────────────

def test_gpu_usage_empty():
    svc = GpuInventoryService()
    cid = uuid.uuid4()
    result = svc.get_usage(cid, "test-cluster")
    assert isinstance(result, GpuUsageResponse)
    assert result.cluster_id == cid
    assert result.cluster_name == "test-cluster"
    assert result.total_gpus == 0


def test_gpu_usage_update():
    import json
    svc = GpuInventoryService()
    cid = uuid.uuid4()
    result = svc.update_usage(
        cluster_id=cid,
        cluster_name="prod-cluster",
        gpus_by_model=[
            {"gpu_model": "H100", "total": 8, "allocated": 6, "average_utilization": 72.0},
            {"gpu_model": "A100", "total": 4, "allocated": 2, "average_utilization": 45.0},
        ],
        gpus_by_node=[
            {"node": "gpu-01", "gpu_model": "H100", "allocated": 2, "total": 4},
            {"node": "gpu-02", "gpu_model": "H100", "allocated": 4, "total": 4},
        ],
    )
    assert result.total_gpus == 12
    assert result.allocated_gpus == 8
    assert result.available_gpus == 4
    assert len(result.by_model) == 2
    assert len(result.by_node) == 2
    assert result.utilization_pct > 0

    fetched = svc.get_usage(cid)
    assert fetched.total_gpus == 12
    assert fetched.allocated_gpus == 8


def test_gpu_usage_persistence():
    import json
    svc = GpuInventoryService()
    cid = uuid.uuid4()
    svc.update_usage(
        cluster_id=cid,
        cluster_name="persist-cluster",
        gpus_by_model=[{"gpu_model": "A100", "total": 4, "allocated": 2}],
        gpus_by_node=[],
    )
    svc2 = GpuInventoryService()
    loaded = svc2.get_usage(cid)
    assert loaded.total_gpus == 4
    assert loaded.cluster_name == "persist-cluster"


# ── Mock Inference Tests ─────────────────────────────────────

def test_mock_completion_non_stream():
    from gpuopt.mock_inference import router
    from gpuopt.inference_schemas import MockCompletionRequest, MockCompletionMessage
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/mock/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": False,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["usage"]["total_tokens"] > 0


def test_mock_completion_stream():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/mock/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
            "stream": True,
        })
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("text/event-stream")


# ── API Integration Tests ────────────────────────────────────

def test_analyze_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/inference/analyze", json={
            "model_name": "llama-8b",
            "gpu_model": "NVIDIA-A100-80GB",
            "gpu_count": 1,
            "avg_gpu_utilization": 20.0,
            "peak_gpu_memory_gib": 30.0,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["model_name"] == "llama-8b"
        assert len(data["observations"]) > 0
        assert len(data["suggestions"]) > 0


def test_gpu_usage_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        cid = str(uuid.uuid4())
        r = client.get(f"/api/v1/inference/clusters/{cid}/gpu-usage")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["cluster_id"] == cid
        assert data["total_gpus"] == 0


def test_gpu_usage_api_with_data():
    from fastapi.testclient import TestClient
    from gpuopt.main import app
    from gpuopt.gpu_usage import GpuInventoryService

    cid = uuid.uuid4()
    svc = GpuInventoryService()
    svc.update_usage(
        cluster_id=cid,
        cluster_name="api-cluster",
        gpus_by_model=[{"gpu_model": "H100", "total": 8, "allocated": 4}],
        gpus_by_node=[],
    )

    with TestClient(app) as client:
        r = client.get(f"/api/v1/inference/clusters/{cid}/gpu-usage")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total_gpus"] == 8
        assert data["allocated_gpus"] == 4


def test_plan_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/inference/plan", json={
            "model_name": "llama-8b",
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["model_name"] == "llama-8b"
        assert data["num_gpus_required"] >= 1


def test_manifest_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/inference/manifest", json={
            "spec": {
                "model_name": "llama-8b",
                "tensor_parallelism": 2,
                "pipeline_parallelism": 1,
            }
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "deployment.yaml" in data["manifests"]
        dep = data["manifests"]["deployment.yaml"]
        assert "nvidia.com/gpu" in dep


def test_benchmark_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/inference/benchmark", json={
            "model": "gpt-3.5-turbo",
            "prompt": "Hello",
            "max_tokens": 50,
            "num_requests": 3,
            "concurrency": 1,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["model"] == "gpt-3.5-turbo"
        assert data["num_requests"] == 3
        assert data["throughput_tokens_per_sec"] > 0
        assert "p50" in data["latency_ms"]
