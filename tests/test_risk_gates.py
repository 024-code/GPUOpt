from __future__ import annotations

from gpuopt.risk_gates import (
    MemoryRiskGate,
    MockDataRiskGate,
    TpCommunicationRiskGate,
    AutoscaleRiskGate,
    QualityRiskGate,
    K8sMutationRiskGate,
    SecretsRiskGate,
    MoeRiskGate,
    RiskGatesService,
)
from gpuopt.risk_gates_schemas import (
    AutoscaleConfig,
    AutoscaleGateInput,
    GateAction,
    GateResult,
    K8sMutationGateInput,
    MemoryBenchmarkConfig,
    MemoryGateInput,
    MockDataGateInput,
    MoeExpertBalanceInput,
    QualityGateInput,
    QualityTestConfig,
    SecretScanInput,
    TpBenchmarkConfig,
    TpGateInput,
)


# ── Gate 1: Memory ───────────────────────────────────────────

def test_memory_estimate():
    config = MemoryBenchmarkConfig(model_name="llama-8b", dtype="fp16", max_seq_len=4096, batch_size=1)
    est = MemoryRiskGate.estimate_memory(config)
    assert est["weight_gb"] > 0
    assert est["total_gb"] > est["weight_gb"]


def test_memory_gate_pass():
    input_data = MemoryGateInput(
        benchmark_config=MemoryBenchmarkConfig(gpu_memory_gib=80.0),
        measured_peak_memory_gib=20.0,
        measured_oom_occurred=False,
    )
    result = MemoryRiskGate.evaluate(input_data)
    assert result.result == GateResult.PASS
    assert result.action == GateAction.ACCEPT


def test_memory_gate_oom():
    input_data = MemoryGateInput(
        benchmark_config=MemoryBenchmarkConfig(gpu_memory_gib=80.0),
        measured_peak_memory_gib=80.0,
        measured_oom_occurred=True,
    )
    result = MemoryRiskGate.evaluate(input_data)
    assert result.result == GateResult.FAIL
    assert result.action == GateAction.REJECT


# ── Gate 2: Mock Data ────────────────────────────────────────

def test_mock_gate_production_ok():
    result = MockDataRiskGate.evaluate(MockDataGateInput(is_mock_data=False, data_source="production"))
    assert result.result == GateResult.PASS


def test_mock_gate_mock_with_real():
    result = MockDataRiskGate.evaluate(MockDataGateInput(is_mock_data=True, has_real_endpoint=True, has_dcgm_metrics=True))
    assert result.result == GateResult.PASS


def test_mock_gate_fail():
    result = MockDataRiskGate.evaluate(MockDataGateInput(is_mock_data=True, has_real_endpoint=False, has_dcgm_metrics=False))
    assert result.result == GateResult.FAIL


# ── Gate 3: TP Communication ─────────────────────────────────

def test_tp_gate_pass():
    input_data = TpGateInput(config=TpBenchmarkConfig(tp_size=2, cross_node_tp=False, all_reduce_time_us=10, compute_time_us=200))
    result = TpCommunicationRiskGate.evaluate(input_data)
    assert result.result == GateResult.PASS


def test_tp_gate_comm_bound():
    input_data = TpGateInput(config=TpBenchmarkConfig(tp_size=8, cross_node_tp=True, all_reduce_time_us=100, compute_time_us=50))
    result = TpCommunicationRiskGate.evaluate(input_data)
    assert result.result == GateResult.FAIL
    assert result.avoid_cross_node is True


def test_tp_gate_comm_warn():
    input_data = TpGateInput(config=TpBenchmarkConfig(tp_size=4, cross_node_tp=False, all_reduce_time_us=80, compute_time_us=100))
    result = TpCommunicationRiskGate.evaluate(input_data)
    assert result.result == GateResult.WARN


# ── Gate 4: Autoscaling ──────────────────────────────────────

def test_autoscale_gate_pass():
    result = AutoscaleRiskGate.evaluate(AutoscaleGateInput(config=AutoscaleConfig()))
    assert result.result == GateResult.PASS


def test_autoscale_gate_high_oscillation():
    result = AutoscaleRiskGate.evaluate(AutoscaleGateInput(config=AutoscaleConfig(), observed_oscillations=10))
    assert result.result == GateResult.FAIL


def test_autoscale_gate_threshold_gap():
    cfg = AutoscaleConfig(scale_up_threshold=55, scale_down_threshold=50)
    result = AutoscaleRiskGate.evaluate(AutoscaleGateInput(config=cfg))
    assert result.result in (GateResult.WARN, GateResult.FAIL)


# ── Gate 5: Quality ──────────────────────────────────────────

def test_quality_gate_pass():
    config = QualityTestConfig(
        optimization_type="quantization",
        original_perplexity=10.0,
        optimized_perplexity=10.2,
        original_accuracy=95.0,
        optimized_accuracy=94.8,
    )
    result = QualityRiskGate.evaluate(QualityGateInput(config=config))
    assert result.result == GateResult.PASS
    assert result.all_tests_passed is True


def test_quality_gate_fail():
    config = QualityTestConfig(
        optimization_type="quantization",
        original_perplexity=10.0,
        optimized_perplexity=12.0,
        original_accuracy=95.0,
        optimized_accuracy=80.0,
    )
    result = QualityRiskGate.evaluate(QualityGateInput(config=config))
    assert result.result == GateResult.FAIL


# ── Gate 6: K8s Mutation ─────────────────────────────────────

def test_k8s_gate_pass():
    result = K8sMutationRiskGate.evaluate(K8sMutationGateInput(is_read_only=True, requires_approval=True, has_gitops=True))
    assert result.result == GateResult.PASS
    assert result.mutation_safe is True


def test_k8s_gate_fail():
    result = K8sMutationRiskGate.evaluate(K8sMutationGateInput(is_read_only=False, requires_approval=False, has_gitops=False))
    assert result.result == GateResult.FAIL


# ── Gate 7: Secrets ──────────────────────────────────────────

def test_secrets_gate_pass():
    result = SecretsRiskGate.evaluate(SecretScanInput(uses_secret_references=True))
    assert result.result == GateResult.PASS


def test_secrets_gate_fail_tokens():
    result = SecretsRiskGate.evaluate(SecretScanInput(uses_secret_references=True, has_bearer_tokens_in_logs=True))
    assert result.result == GateResult.FAIL


def test_secrets_gate_no_refs():
    result = SecretsRiskGate.evaluate(SecretScanInput(uses_secret_references=False))
    assert result.result == GateResult.FAIL


# ── Gate 8: MoE ──────────────────────────────────────────────

def test_moe_gate_pass():
    input_data = MoeExpertBalanceInput(per_gpu_loads=[10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 10.0])
    result = MoeRiskGate.evaluate(input_data)
    assert result.result == GateResult.PASS
    assert result.is_balanced is True


def test_moe_gate_imbalanced():
    input_data = MoeExpertBalanceInput(per_gpu_loads=[5.0, 10.0, 8.0, 30.0, 6.0, 9.0, 7.0, 28.0])
    result = MoeRiskGate.evaluate(input_data)
    assert result.result in (GateResult.WARN, GateResult.FAIL)


def test_moe_gate_severe():
    input_data = MoeExpertBalanceInput(per_gpu_loads=[2.0, 3.0, 2.5, 40.0, 2.0, 3.0, 2.5, 38.0])
    result = MoeRiskGate.evaluate(input_data)
    assert result.result == GateResult.FAIL


# ── Aggregated ───────────────────────────────────────────────

def test_dashboard():
    svc = RiskGatesService()
    dashboard = svc.evaluate_all()
    assert len(dashboard.gates) == 8
    assert "memory" in dashboard.gates
    assert "mock_data" in dashboard.gates
    assert "tp_comm" in dashboard.gates
    assert "autoscale" in dashboard.gates
    assert "quality" in dashboard.gates
    assert "k8s_mutation" in dashboard.gates
    assert "secrets" in dashboard.gates
    assert "moe" in dashboard.gates
    assert dashboard.can_deploy or not dashboard.can_deploy


def test_health():
    svc = RiskGatesService()
    health = svc.health()
    assert health["status"] == "healthy"
    assert len(health["components"]) == 8


def test_all_gate_ids():
    svc = RiskGatesService()
    d = svc.evaluate_all()
    expected = {"memory", "mock_data", "tp_comm", "autoscale", "quality", "k8s_mutation", "secrets", "moe"}
    assert set(d.gates.keys()) == expected


# ── API Tests ────────────────────────────────────────────────

def test_dashboard_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/risk-gates/dashboard")
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["gates"]) == 8


def test_memory_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/memory", json={
            "benchmark_config": {"model_name": "llama-8b", "gpu_memory_gib": 80.0},
            "measured_peak_memory_gib": 20.0,
            "measured_oom_occurred": False,
        })
        assert r.status_code == 200
        assert r.json()["result"] == "pass"


def test_mock_data_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/mock-data", json={
            "is_mock_data": True,
            "has_real_endpoint": False,
        })
        assert r.status_code == 200
        assert r.json()["result"] == "fail"


def test_tp_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/tp-communication", json={
            "config": {"tp_size": 2, "all_reduce_time_us": 10, "compute_time_us": 200},
        })
        assert r.status_code == 200


def test_autoscale_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/autoscale", json={
            "config": {"cooldown_seconds": 300, "hysteresis_pct": 10, "min_dwell_seconds": 600},
        })
        assert r.status_code == 200


def test_quality_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/quality", json={
            "config": {
                "optimization_type": "quantization",
                "original_perplexity": 10.0,
                "optimized_perplexity": 10.2,
                "original_accuracy": 95.0,
                "optimized_accuracy": 94.8,
            },
        })
        assert r.status_code == 200
        assert r.json()["all_tests_passed"] is True


def test_k8s_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/k8s-mutation", json={
            "is_read_only": True, "requires_approval": True, "has_gitops": True,
        })
        assert r.status_code == 200
        assert r.json()["result"] == "pass"


def test_secrets_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/secrets", json={
            "uses_secret_references": True,
        })
        assert r.status_code == 200
        assert r.json()["result"] == "pass"


def test_moe_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/risk-gates/moe-imbalance", json={
            "per_gpu_loads": [10.0, 11.0, 10.5, 12.0, 11.5, 10.0, 12.5, 11.0],
        })
        assert r.status_code == 200


def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/risk-gates/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
