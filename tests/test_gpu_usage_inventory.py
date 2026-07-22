from __future__ import annotations

from gpuopt.gpu_usage_inventory import GpuUsageInventoryService
from gpuopt.gpu_usage_inventory_schemas import (
    ClusterInventory,
    DcgmTelemetryResponse,
    GpuInventorySnapshot,
    GpuTelemetrySample,
    NodeGpuAllocation,
)


# ── Unit: Service ─────────────────────────────────────────────

def test_get_inventory():
    svc = GpuUsageInventoryService()
    inv = svc.get_inventory()
    assert isinstance(inv, ClusterInventory)
    assert inv.snapshot.total_gpu_capacity == 8
    assert inv.snapshot.allocated_to_all_workloads == 6
    assert inv.snapshot.estimated_free == 2
    assert inv.snapshot.allocation_utilization_pct == 75.0
    assert len(inv.nodes) == 3
    assert "K8s" in inv.snapshot.summary


def test_get_inventory_custom_cluster():
    svc = GpuUsageInventoryService()
    inv = svc.get_inventory("prod-cluster")
    assert inv.cluster_id == "prod-cluster"
    assert all("prod-cluster" in n.node_name for n in inv.nodes)


def test_inventory_node_allocation():
    svc = GpuUsageInventoryService()
    inv = svc.get_inventory()
    inference_nodes = [n for n in inv.nodes if n.allocated_to == "inference"]
    assert len(inference_nodes) >= 1
    assert len(inference_nodes[0].pods) == 2
    assert inference_nodes[0].gpu_model == "NVIDIA H100-SXM-80GB"


def test_inventory_free_gpus():
    svc = GpuUsageInventoryService()
    inv = svc.get_inventory()
    free_nodes = [n for n in inv.nodes if n.allocated_to == "free"]
    assert len(free_nodes) == 1
    assert free_nodes[0].pods == []


def test_inventory_dcgm_message():
    svc = GpuUsageInventoryService()
    inv = svc.get_inventory()
    assert "DCGM exporter" in inv.dcgm_required_message


def test_dcgm_telemetry_default():
    svc = GpuUsageInventoryService()
    tele = svc.get_dcgm_telemetry()
    assert isinstance(tele, DcgmTelemetryResponse)
    assert tele.daemonset_running is True
    assert len(tele.samples) == 8
    assert tele.samples[0].gpu_index == 0
    assert tele.samples[0].engine_util_pct > 0
    assert tele.samples[0].framebuffer_total_gib == 80.0


def test_dcgm_telemetry_custom_gpus():
    svc = GpuUsageInventoryService()
    tele = svc.get_dcgm_telemetry(num_gpus=2)
    assert len(tele.samples) == 2


def test_dcgm_telemetry_summary():
    svc = GpuUsageInventoryService()
    tele = svc.get_dcgm_telemetry(cluster_id="test-cluster", num_gpus=4)
    assert tele.cluster_id == "test-cluster"
    assert "DCGM" in tele.summary
    assert "allocation != utilization" in tele.summary


def test_health():
    svc = GpuUsageInventoryService()
    h = svc.health()
    assert h["status"] == "healthy"


# ── API Tests ─────────────────────────────────────────────────

def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/gpu-usage-inventory/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


def test_cluster_inventory_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/gpu-usage-inventory/cluster")
        assert r.status_code == 200
        data = r.json()
        assert data["snapshot"]["total_gpu_capacity"] == 8
        assert len(data["nodes"]) == 3


def test_cluster_inventory_api_with_id():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/gpu-usage-inventory/cluster?cluster_id=prod-east")
        assert r.status_code == 200
        assert r.json()["cluster_id"] == "prod-east"


def test_dcgm_telemetry_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/gpu-usage-inventory/dcgm-telemetry?num_gpus=4")
        assert r.status_code == 200
        data = r.json()
        assert len(data["samples"]) == 4
        assert data["daemonset_running"] is True
