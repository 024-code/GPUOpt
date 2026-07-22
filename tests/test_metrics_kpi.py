from __future__ import annotations

from gpuopt.metrics_kpi import (
    RequestMetricsCollector,
    GpuMetricsCollector,
    ReliabilityMetricsCollector,
    ThermalMetricsCollector,
    PlacementMetricsCollector,
    EconomicsMetricsCollector,
    MetricsKpiDashboardService,
)
from gpuopt.metrics_kpi_schemas import (
    MetricsDashboard,
    RequestMetrics,
    GpuMetricsResult,
    ReliabilityMetrics,
    ThermalMetrics,
    PlacementMetrics,
    EconomicsMetrics,
)


# ── Layer 1: Request ─────────────────────────────────────────

def test_request_collect():
    metrics = RequestMetricsCollector.collect(arrival_rate=100.0)
    assert isinstance(metrics, RequestMetrics)
    assert metrics.arrival_rate_req_per_sec == 100.0
    assert metrics.latency_p50_ms > 0
    assert metrics.ttft_ms_avg > 0
    assert metrics.total_requests > 0


def test_request_decide_slo_ok():
    metrics = RequestMetrics(
        arrival_rate_req_per_sec=50.0,
        prompt_tokens_avg=1024,
        output_tokens_avg=256,
        queue_time_ms_avg=10,
        ttft_ms_avg=80,
        tpot_ms_avg=20,
        latency_p50_ms=100,
        latency_p95_ms=200,
        latency_p99_ms=300,
        error_rate=0.001,
    )
    dec = RequestMetricsCollector.decide(metrics)
    assert dec.slo_breached is False
    assert dec.recommended_replicas >= 1


def test_request_decide_slo_breached():
    metrics = RequestMetrics(
        arrival_rate_req_per_sec=50.0, prompt_tokens_avg=1024, output_tokens_avg=256,
        queue_time_ms_avg=10, ttft_ms_avg=80, tpot_ms_avg=20,
        latency_p50_ms=100, latency_p95_ms=1000, latency_p99_ms=3000,
        error_rate=0.001,
    )
    dec = RequestMetricsCollector.decide(metrics)
    assert dec.slo_breached is True


# ── Layer 2: GPU ─────────────────────────────────────────────

def test_gpu_collect():
    result = GpuMetricsCollector.collect(num_gpus=4)
    assert isinstance(result, GpuMetricsResult)
    assert len(result.samples) == 4
    assert result.avg_engine_util > 0
    assert result.total_framebuffer_used_gib > 0


def test_gpu_decide_compute_bound():
    samples = [GpuMetricsCollector.collect(1)]
    metrics = GpuMetricsCollector.collect(1)
    # force high engine, low dram
    metrics.avg_engine_util = 90.0
    metrics.avg_dram_activity = 20.0
    dec = GpuMetricsCollector.decide(metrics)
    assert dec.bottleneck == "compute"


def test_gpu_decide_memory_bound():
    metrics = GpuMetricsCollector.collect(1)
    metrics.avg_engine_util = 30.0
    metrics.avg_dram_activity = 80.0
    dec = GpuMetricsCollector.decide(metrics)
    assert dec.bottleneck == "memory"


# ── Layer 3: Reliability ─────────────────────────────────────

def test_reliability_collect():
    metrics = ReliabilityMetricsCollector.collect()
    assert isinstance(metrics, ReliabilityMetrics)
    assert metrics.oom_count >= 0
    assert len(metrics.incidents) == 5


def test_reliability_decide_critical():
    metrics = ReliabilityMetrics(oom_count=5, xid_error_count=3, pod_restart_count=4, retry_count=0, failed_request_count=0, incidents=[])
    dec = ReliabilityMetricsCollector.decide(metrics)
    assert dec.priority == "critical"
    assert dec.requires_rollback is True


def test_reliability_decide_healthy():
    metrics = ReliabilityMetrics(oom_count=0, xid_error_count=0, pod_restart_count=0, retry_count=0, failed_request_count=0, incidents=[])
    dec = ReliabilityMetricsCollector.decide(metrics)
    assert dec.priority == "low"


# ── Layer 4: Thermal ─────────────────────────────────────────

def test_thermal_collect():
    metrics = ThermalMetricsCollector.collect(num_gpus=4)
    assert isinstance(metrics, ThermalMetrics)
    assert metrics.gpu_temp_celsius_avg > 0
    assert metrics.power_draw_watts_avg > 0


def test_thermal_decide_hot():
    metrics = ThermalMetrics(
        gpu_temp_celsius_avg=80.0, gpu_temp_celsius_max=92.0,
        memory_temp_celsius_avg=75.0, power_draw_watts_avg=350.0,
        power_draw_watts_max=400.0, power_limit_watts=400.0,
        throttling_active=True, throttling_reason="Temp > 88C",
    )
    dec = ThermalMetricsCollector.decide(metrics)
    assert dec.requires_cooling_action is True
    assert dec.requires_rescheduling is True


def test_thermal_decide_normal():
    metrics = ThermalMetrics(
        gpu_temp_celsius_avg=55.0, gpu_temp_celsius_max=62.0,
        memory_temp_celsius_avg=58.0, power_draw_watts_avg=200.0,
        power_draw_watts_max=250.0, power_limit_watts=400.0,
        throttling_active=False,
    )
    dec = ThermalMetricsCollector.decide(metrics)
    assert dec.requires_cooling_action is False
    assert dec.recommended_power_cap_watts == 400.0


# ── Layer 5: Placement ───────────────────────────────────────

def test_placement_collect():
    metrics = PlacementMetricsCollector.collect(num_nodes=2)
    assert isinstance(metrics, PlacementMetrics)
    assert len(metrics.nodes) == 2
    assert metrics.avg_gpu_utilization > 0


def test_placement_decide():
    metrics = PlacementMetricsCollector.collect(3)
    dec = PlacementMetricsCollector.decide(metrics)
    assert dec.recommended_tensor_parallelism >= 1
    assert dec.recommended_pipeline_parallelism >= 1
    assert dec.placement_strategy in ("consolidate", "balance", "scale_out")


# ── Layer 6: Economics ───────────────────────────────────────

def test_economics_collect():
    metrics = EconomicsMetricsCollector.collect(gpu_count=32, hourly_cost_per_gpu=1.5, tokens_per_sec=5000.0)
    assert isinstance(metrics, EconomicsMetrics)
    assert metrics.total_gpu_hours > 0
    assert metrics.cost_per_million_tokens > 0


def test_economics_decide():
    metrics = EconomicsMetrics(
        total_gpu_hours=23360.0, tokens_per_gpu_second=156.25,
        cost_per_million_tokens=0.085, idle_gpu_hours=5000.0,
        reserved_gpu_hours=3000.0, total_cost_usd=35040.0,
        potential_savings_usd=5250.0, utilization_effective_pct=78.6,
    )
    dec = EconomicsMetricsCollector.decide(metrics)
    assert dec.estimated_savings_usd > 0


# ── Aggregated Dashboard ─────────────────────────────────────

def test_dashboard():
    svc = MetricsKpiDashboardService()
    dashboard = svc.build_dashboard(gpu_count=8, num_nodes=2)
    assert isinstance(dashboard, MetricsDashboard)
    assert dashboard.request.arrival_rate_req_per_sec > 0
    assert len(dashboard.gpu.samples) == 8
    assert len(dashboard.reliability.incidents) == 5
    assert dashboard.thermal.gpu_temp_celsius_avg > 0
    assert len(dashboard.placement.nodes) == 2
    assert dashboard.economics.total_gpu_hours > 0
    assert len(dashboard.layer_statuses) == 6


def test_dashboard_statuses():
    svc = MetricsKpiDashboardService()
    dashboard = svc.build_dashboard()
    layers = {s.layer: s.status for s in dashboard.layer_statuses}
    assert "request" in layers
    assert "gpu" in layers
    assert "reliability" in layers
    assert "thermal" in layers
    assert "placement" in layers
    assert "economics" in layers


def test_health():
    svc = MetricsKpiDashboardService()
    health = svc.health()
    assert health["status"] == "healthy"
    assert len(health["components"]) == 6


# ── API Tests ────────────────────────────────────────────────

def test_dashboard_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/dashboard?gpu_count=4&num_nodes=2")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "request" in data
        assert "gpu" in data
        assert "reliability" in data
        assert "thermal" in data
        assert "placement" in data
        assert "economics" in data
        assert len(data["layer_statuses"]) == 6


def test_request_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/request")
        assert r.status_code == 200
        data = r.json()
        assert "metrics" in data
        assert "decision" in data


def test_gpu_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/gpu?num_gpus=4")
        assert r.status_code == 200
        data = r.json()
        assert "metrics" in data
        assert "decision" in data


def test_reliability_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/reliability")
        assert r.status_code == 200
        data = r.json()
        assert "metrics" in data


def test_thermal_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/thermal?num_gpus=4")
        assert r.status_code == 200


def test_placement_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/placement?num_nodes=3")
        assert r.status_code == 200
        data = r.json()
        assert len(data["metrics"]["nodes"]) == 3


def test_economics_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/economics?gpu_count=16")
        assert r.status_code == 200


def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/metrics-kpi/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
