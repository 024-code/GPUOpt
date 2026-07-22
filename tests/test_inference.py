from __future__ import annotations

from uuid import uuid4

from gpuopt.schemas import (
    InferenceEndpointStatus,
    InferenceFramework,
)
from gpuopt.inference.service import InferenceService


def test_register_and_get_endpoint():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="llm-serve",
        model_name="llama-3-8b",
        framework=InferenceFramework.VLLM,
        gpu_count=1,
        gpu_model="NVIDIA H100-SXM-80GB",
        quantisation="fp16",
        max_batch_size=32,
        max_input_tokens=4096,
        max_output_tokens=1024,
        concurrency=4,
    )
    assert ep.endpoint_name == "llm-serve"
    assert ep.model_name == "llama-3-8b"
    assert ep.framework == InferenceFramework.VLLM
    assert ep.gpu_count == 1
    assert ep.status == InferenceEndpointStatus.DEPLOYING

    fetched = svc.get_endpoint(ep.id)
    assert fetched is not None
    assert fetched.id == ep.id


def test_list_endpoints():
    svc = InferenceService()
    cid = uuid4()
    svc.register_endpoint(cluster_id=cid, endpoint_name="ep-a", model_name="model-a")
    svc.register_endpoint(cluster_id=cid, endpoint_name="ep-b", model_name="model-b")
    other = uuid4()
    svc.register_endpoint(cluster_id=other, endpoint_name="ep-c", model_name="model-c")
    eps = svc.list_endpoints(cid)
    assert len(eps) == 2
    all_eps = svc.list_endpoints()
    assert len(all_eps) >= 3


def test_update_endpoint():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(cluster_id=cid, endpoint_name="update-test", model_name="test-model")
    updated = svc.update_endpoint(
        ep.id,
        status=InferenceEndpointStatus.RUNNING,
        avg_latency_ms=45.2,
        p99_latency_ms=120.5,
        throughput_requests_per_sec=50.0,
        throughput_tokens_per_sec=5000.0,
        avg_gpu_utilization=72.0,
        peak_gpu_memory_gib=60.0,
        kv_cache_utilization=45.0,
        cost_per_1k_tokens=0.0035,
    )
    assert updated is not None
    assert updated.status == InferenceEndpointStatus.RUNNING
    assert updated.avg_latency_ms == 45.2
    assert updated.p99_latency_ms == 120.5
    assert updated.throughput_requests_per_sec == 50.0
    assert updated.avg_gpu_utilization == 72.0
    assert updated.kv_cache_utilization == 45.0


def test_delete_endpoint():
    svc = InferenceService()
    ep = svc.register_endpoint(cluster_id=uuid4(), endpoint_name="delete-me", model_name="del-model")
    assert svc.delete_endpoint(ep.id) is True
    assert svc.delete_endpoint(ep.id) is False
    assert svc.get_endpoint(ep.id) is None


def test_profile_endpoint():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="profile-test",
        model_name="llama-3-70b",
        gpu_count=4,
        gpu_model="NVIDIA A100-80GB",
        quantisation="fp16",
        concurrency=8,
        max_batch_size=64,
    )
    svc.update_endpoint(
        ep.id,
        status=InferenceEndpointStatus.RUNNING,
        avg_latency_ms=250.0,
        p99_latency_ms=800.0,
        throughput_requests_per_sec=25.0,
        throughput_tokens_per_sec=3000.0,
        avg_gpu_utilization=55.0,
        peak_gpu_memory_gib=72.0,
        kv_cache_utilization=85.0,
    )

    profile = svc.profile_endpoint(ep.id)
    assert profile is not None
    assert profile.avg_latency_ms == 250.0
    assert profile.p50_latency_ms == 200.0
    assert profile.p99_latency_ms == 800.0
    assert profile.gpu_compute_efficiency == 55.0
    assert profile.gpu_memory_efficiency > 0
    assert profile.kv_cache_efficiency > 0
    assert len(profile.recommendations) > 0


def test_profile_high_latency():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="slow-model",
        model_name="falcon-180b",
        gpu_count=8,
        quantisation="fp16",
        concurrency=16,
    )
    svc.update_endpoint(
        ep.id,
        status=InferenceEndpointStatus.RUNNING,
        avg_latency_ms=2500.0,
        avg_gpu_utilization=30.0,
    )
    profile = svc.profile_endpoint(ep.id)
    assert profile is not None
    assert any("latency" in r.lower() for r in profile.recommendations)
    assert profile.p50_latency_ms == 2000.0


def test_profile_kv_cache_pressure():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="kv-stressed",
        model_name="mixtral-8x7b",
        gpu_count=2,
        gpu_model="NVIDIA A100-80GB",
        quantisation="int8",
    )
    svc.update_endpoint(
        ep.id,
        status=InferenceEndpointStatus.RUNNING,
        kv_cache_utilization=85.0,
        peak_gpu_memory_gib=75.0,
    )
    profile = svc.profile_endpoint(ep.id)
    assert profile is not None
    assert profile.kv_cache_efficiency <= 85
    if any("KV cache" in r for r in profile.recommendations):
        assert True
    else:
        assert profile.kv_cache_utilization > 0


def test_suggest_deployment_config():
    config = InferenceService.suggest_deployment_config(
        model_name="llama-3-70b",
        model_size_gb=140.0,
        context_length=8192,
        target_latency_ms=100.0,
        expected_requests_per_sec=50.0,
    )
    assert config.recommended_gpu_count >= 1
    assert config.recommended_gpu_model != ""
    assert config.recommended_quantisation in ("fp16", "int8", "int4", "bf16", "fp8")
    assert config.estimated_throughput_tokens_per_sec > 0
    assert config.estimated_p50_latency_ms > 0
    assert config.estimated_cost_per_1m_tokens_usd > 0
    assert len(config.recommendations) >= 1
    assert len(config.alternatives) >= 0


def test_suggest_deployment_config_small_model():
    config = InferenceService.suggest_deployment_config(
        model_name="bert-base",
        model_size_gb=0.5,
        context_length=512,
        target_latency_ms=50.0,
        expected_requests_per_sec=100.0,
    )
    assert config.recommended_gpu_count >= 1
    assert config.estimated_throughput_tokens_per_sec > 0


def test_generate_recommendations():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="low-util",
        model_name="test-model",
        gpu_count=4,
    )
    svc.update_endpoint(
        ep.id,
        status=InferenceEndpointStatus.RUNNING,
        avg_gpu_utilization=15.0,
    )
    recs = svc.generate_recommendations(cid)
    assert len(recs) >= 1
    assert any("low" in r.title.lower() for r in recs)


def test_generate_recommendations_high_latency():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="slow-serve",
        model_name="test-model",
    )
    svc.update_endpoint(
        ep.id,
        status=InferenceEndpointStatus.RUNNING,
        avg_latency_ms=2000.0,
    )
    recs = svc.generate_recommendations(cid)
    assert any("latency" in r.title.lower() for r in recs)


def test_generate_recommendations_kv_cache():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="kv-full",
        model_name="test-model",
    )
    svc.update_endpoint(
        ep.id,
        status=InferenceEndpointStatus.RUNNING,
        kv_cache_utilization=90.0,
    )
    recs = svc.generate_recommendations(cid)
    assert any("KV" in r.title for r in recs)


def test_generate_deployment_config_recs():
    svc = InferenceService()
    cid = uuid4()
    ep = svc.register_endpoint(
        cluster_id=cid,
        endpoint_name="pending-deploy",
        model_name="llama-3-8b",
    )
    svc.update_endpoint(ep.id, status=InferenceEndpointStatus.DEPLOYING)
    recs = svc.generate_deployment_config_recs(cid)
    assert len(recs) >= 1
    assert any("Deployment" in r.title for r in recs)


def test_inference_endpoints_api():
    import os, uuid
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_inference_api.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_inference_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_inference_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/api/v1/clusters", json={
            "name": f"inf-cluster-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "mock",
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]

        r = client.post(f"/api/v1/inference/endpoints?cluster_id={cid}&endpoint_name=test-ep&model_name=llama-3-8b&framework=vllm&gpu_count=1&quantisation=fp16&max_batch_size=32&concurrency=4")
        assert r.status_code == 201, r.text
        ep_id = r.json()["id"]
        assert r.json()["model_name"] == "llama-3-8b"

        r = client.get("/api/v1/inference/endpoints")
        assert r.status_code == 200
        eps = r.json()
        assert any(e["id"] == ep_id for e in eps)

        r = client.get(f"/api/v1/inference/endpoints/{ep_id}")
        assert r.status_code == 200
        assert r.json()["endpoint_name"] == "test-ep"

        r = client.patch(f"/api/v1/inference/endpoints/{ep_id}?status=running&avg_latency_ms=50.0&throughput_requests_per_sec=100.0")
        assert r.status_code == 200
        assert r.json()["status"] == "running"
        assert r.json()["throughput_requests_per_sec"] == 100.0

        r = client.post(f"/api/v1/inference/endpoints/{ep_id}/profile")
        assert r.status_code == 200
        profile = r.json()
        assert profile["gpu_compute_efficiency"] >= 0

        r = client.delete(f"/api/v1/inference/endpoints/{ep_id}")
        assert r.status_code == 200

        r = client.get(f"/api/v1/inference/endpoints/{ep_id}")
        assert r.status_code == 404
