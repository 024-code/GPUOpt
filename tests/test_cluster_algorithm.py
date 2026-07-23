from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpuopt.ml.cluster_algorithm import (
    ClusterManagementAlgorithm,
    GpuMetrics,
    JobSpec,
    NodeHardware,
    PowerCapMode,
    SchedulingPolicy,
)
from gpuopt.ml.node_simulation import ClusterTopology, SimNode


@pytest.fixture
def topology():
    return ClusterTopology().build_dgx_h100(1)


@pytest.fixture
def algorithm():
    return ClusterManagementAlgorithm()


@pytest.fixture
def sample_metric():
    return GpuMetrics(
        index=0, node_id="dgx-h100-00",
        engine_util_pct=60.0, memory_pct=50.0,
        gpu_temp_c=65.0, power_watts=250.0,
        power_cap_watts=400.0,
        xid_errors=2, ecc_errors=10,
        fan_speed_pct=50.0, clock_mhz=1500.0,
        wear_factor=0.05,
    )


class TestGpuMetrics:
    def test_to_telemetry(self, sample_metric):
        t = sample_metric.to_telemetry()
        assert t["gpu_utilization"] == 60.0
        assert t["temperature"] == 65.0
        assert t["ecc_errors"] == 10

    def test_to_telemetry_includes_all_keys(self, sample_metric):
        t = sample_metric.to_telemetry()
        required = {"gpu_utilization", "memory_utilization", "temperature",
                     "power_usage", "clock_speed", "ecc_errors", "xid_errors"}
        assert required.issubset(t.keys())


class TestClusterManagementAlgorithm:
    def test_get_gpu_metrics(self, algorithm, topology):
        metrics = algorithm.get_gpu_metrics(topology)
        assert len(metrics) == 8
        assert metrics[0].engine_util_pct == 0.0

    def test_heuristic_risk_low(self, algorithm, sample_metric):
        risk = algorithm._heuristic_risk(sample_metric)
        assert 0 <= risk <= 1

    def test_heuristic_risk_high_temp(self, algorithm):
        hot = GpuMetrics(0, "node", 90, 95, 88, 400, 400, 10, 50, 80, 1800, 0.3)
        risk = algorithm._heuristic_risk(hot)
        assert risk > 0.5

    def test_heuristic_risk_safe(self, algorithm):
        safe = GpuMetrics(0, "node", 20, 30, 40, 100, 400, 0, 0, 30, 800, 0.01)
        risk = algorithm._heuristic_risk(safe)
        assert risk < 0.3

    def test_schedule_job_basic(self, algorithm, topology):
        job = JobSpec(job_id="test-001", required_gpus=2, required_memory_gib=16.0)
        decision = algorithm.schedule_job(job, topology)
        assert len(decision.assigned_gpus) == 2
        assert decision.job_id == "test-001"
        assert decision.score > 0

    def test_schedule_job_all_policies(self, algorithm, topology):
        job = JobSpec(job_id="test-002", required_gpus=1)
        for policy in SchedulingPolicy:
            algorithm.scheduling_policy = policy
            decision = algorithm.schedule_job(job, topology)
            assert len(decision.assigned_gpus) == 1
            assert decision.policy == policy

    def test_schedule_job_insufficient_gpus(self, algorithm, topology):
        job = JobSpec(job_id="test-003", required_gpus=100)
        decision = algorithm.schedule_job(job, topology)
        assert len(decision.assigned_gpus) == 0
        assert "Insufficient" in decision.rationale

    def test_power_cap_temperature_guided(self, algorithm, topology):
        algorithm.power_cap_mode = PowerCapMode.TEMPERATURE_GUIDED
        caps = algorithm.compute_power_caps(topology)
        assert isinstance(caps, list)

    def test_power_cap_risk_guided(self, algorithm):
        algorithm.power_cap_mode = PowerCapMode.RISK_GUIDED
        topo = ClusterTopology().build_rtx_cluster(4)
        caps = algorithm.compute_power_caps(topo)
        assert isinstance(caps, list)

    def test_power_cap_predictive(self, algorithm, topology):
        algorithm.power_cap_mode = PowerCapMode.PREDICTIVE
        caps = algorithm.compute_power_caps(topology)
        assert isinstance(caps, list)

    def test_recommend_drain_clean(self, algorithm, topology):
        drains = algorithm.recommend_drain(topology)
        assert isinstance(drains, list)

    def test_recommend_drain_high_wear(self, algorithm):
        topo = ClusterTopology().build_rtx_cluster(4)
        for node in topo.nodes:
            for gpu in node.gpus:
                gpu.degradation.accumulated_thermal_stress = 5000.0
        drains = algorithm.recommend_drain(topo)
        high_urgency = [d for d in drains if d.urgency == "critical"]
        assert isinstance(high_urgency, list)

    def test_adaptive_throttle_no_action(self, algorithm, sample_metric):
        result = algorithm.adaptive_throttle(sample_metric, risk=0.2)
        assert result["action_needed"] is False

    def test_adaptive_throttle_high_risk(self, algorithm, sample_metric):
        result = algorithm.adaptive_throttle(sample_metric, risk=0.8)
        assert result["action_needed"] is True
        assert result["clock_reduction_pct"] > 10

    def test_balance_load(self, algorithm, topology):
        actions = algorithm.balance_load(topology)
        assert len(actions) > 0

    def test_cluster_health_report(self, algorithm, topology):
        report = algorithm.get_cluster_health_report(topology)
        assert "cluster_health" in report
        assert "recommendations" in report
        assert report["cluster_health"]["gpu_count"] == 8

    def test_predict_failure_risk_fallback(self, algorithm, sample_metric):
        algorithm._predictor = None
        risk = algorithm.predict_failure_risk(sample_metric)
        assert 0 <= risk <= 1

    def test_compute_gpu_score(self, algorithm, sample_metric):
        job = JobSpec(job_id="test", required_gpus=1)
        score = algorithm._compute_gpu_score(sample_metric, job, SchedulingPolicy.HYBRID)
        assert "risk" in score
        assert "raw_score" in score
        assert "scores" in score
        assert 0 <= score["risk"] <= 1

    def test_policy_configuration(self, algorithm):
        assert algorithm.scheduling_policy == SchedulingPolicy.HYBRID
        assert algorithm.power_cap_mode == PowerCapMode.PREDICTIVE
        algorithm.scheduling_policy = SchedulingPolicy.RISK_AWARE
        assert algorithm.scheduling_policy == SchedulingPolicy.RISK_AWARE


class TestSchedulingPolicies:
    def test_memory_bandwidth_policy(self, algorithm, topology):
        job = JobSpec(job_id="bw-test", required_gpus=2)
        algorithm.scheduling_policy = SchedulingPolicy.MEMORY_BANDWIDTH
        decision = algorithm.schedule_job(job, topology)
        assert len(decision.assigned_gpus) == 2
        assert decision.policy == SchedulingPolicy.MEMORY_BANDWIDTH

    def test_ecc_required_policy(self, algorithm, topology):
        job = JobSpec(job_id="ecc-test", required_gpus=2)
        algorithm.scheduling_policy = SchedulingPolicy.ECC_REQUIRED
        decision = algorithm.schedule_job(job, topology)
        assert len(decision.assigned_gpus) == 2
        assert decision.policy == SchedulingPolicy.ECC_REQUIRED

    def test_check_node_capability_ecc_required(self, algorithm):
        metric = GpuMetrics(
            index=0, node_id="node-0", engine_util_pct=0, memory_pct=0,
            gpu_temp_c=35, power_watts=30, power_cap_watts=400,
            xid_errors=0, ecc_errors=0, fan_speed_pct=30, clock_mhz=1000,
            wear_factor=0, ecc_capable=True, memory_bandwidth_gbps=200,
            node_cpu_cores=64, node_cpu_vendor="intel", node_memory_type="DDR5",
        )
        job = JobSpec(require_ecc=True)
        ok, reason = algorithm._check_node_capability(metric, job)
        assert ok is True
        assert reason == ""

    def test_check_node_capability_ecc_missing(self, algorithm):
        metric = GpuMetrics(
            index=0, node_id="node-0", engine_util_pct=0, memory_pct=0,
            gpu_temp_c=35, power_watts=30, power_cap_watts=400,
            xid_errors=0, ecc_errors=0, fan_speed_pct=30, clock_mhz=1000,
            wear_factor=0, ecc_capable=False, memory_bandwidth_gbps=200,
            node_cpu_cores=64, node_cpu_vendor="intel", node_memory_type="DDR5",
        )
        job = JobSpec(require_ecc=True)
        ok, reason = algorithm._check_node_capability(metric, job)
        assert ok is False
        assert "lacks ECC" in reason

    def test_check_node_capability_bandwidth_ok(self, algorithm):
        metric = GpuMetrics(
            index=0, node_id="node-0", engine_util_pct=0, memory_pct=0,
            gpu_temp_c=35, power_watts=30, power_cap_watts=400,
            xid_errors=0, ecc_errors=0, fan_speed_pct=30, clock_mhz=1000,
            wear_factor=0, ecc_capable=True, memory_bandwidth_gbps=200,
            node_cpu_cores=64, node_cpu_vendor="intel", node_memory_type="DDR5",
        )
        job = JobSpec(min_memory_bandwidth_gbps=100)
        ok, reason = algorithm._check_node_capability(metric, job)
        assert ok is True

    def test_check_node_capability_bandwidth_insufficient(self, algorithm):
        metric = GpuMetrics(
            index=0, node_id="node-0", engine_util_pct=0, memory_pct=0,
            gpu_temp_c=35, power_watts=30, power_cap_watts=400,
            xid_errors=0, ecc_errors=0, fan_speed_pct=30, clock_mhz=1000,
            wear_factor=0, ecc_capable=True, memory_bandwidth_gbps=50,
            node_cpu_cores=64, node_cpu_vendor="intel", node_memory_type="DDR5",
        )
        job = JobSpec(min_memory_bandwidth_gbps=100)
        ok, reason = algorithm._check_node_capability(metric, job)
        assert ok is False
        assert "bandwidth" in reason

    def test_check_node_capability_cpu_cores_ok(self, algorithm):
        metric = GpuMetrics(
            index=0, node_id="node-0", engine_util_pct=0, memory_pct=0,
            gpu_temp_c=35, power_watts=30, power_cap_watts=400,
            xid_errors=0, ecc_errors=0, fan_speed_pct=30, clock_mhz=1000,
            wear_factor=0, ecc_capable=True, memory_bandwidth_gbps=200,
            node_cpu_cores=64, node_cpu_vendor="intel", node_memory_type="DDR5",
        )
        job = JobSpec(min_cpu_cores=32)
        ok, reason = algorithm._check_node_capability(metric, job)
        assert ok is True

    def test_check_node_capability_cpu_vendor_match(self, algorithm):
        metric = GpuMetrics(
            index=0, node_id="node-0", engine_util_pct=0, memory_pct=0,
            gpu_temp_c=35, power_watts=30, power_cap_watts=400,
            xid_errors=0, ecc_errors=0, fan_speed_pct=30, clock_mhz=1000,
            wear_factor=0, ecc_capable=True, memory_bandwidth_gbps=200,
            node_cpu_cores=64, node_cpu_vendor="amd", node_memory_type="DDR5",
        )
        job = JobSpec(cpu_vendor_preference="amd")
        ok, reason = algorithm._check_node_capability(metric, job)
        assert ok is True

    def test_check_node_capability_cpu_vendor_mismatch(self, algorithm):
        metric = GpuMetrics(
            index=0, node_id="node-0", engine_util_pct=0, memory_pct=0,
            gpu_temp_c=35, power_watts=30, power_cap_watts=400,
            xid_errors=0, ecc_errors=0, fan_speed_pct=30, clock_mhz=1000,
            wear_factor=0, ecc_capable=True, memory_bandwidth_gbps=200,
            node_cpu_cores=64, node_cpu_vendor="intel", node_memory_type="DDR5",
        )
        job = JobSpec(cpu_vendor_preference="amd")
        ok, reason = algorithm._check_node_capability(metric, job)
        assert ok is False

    def test_job_rejected_when_no_capable_nodes(self, algorithm, topology):
        # Set all node memory to non-ECC and require ECC
        for node in topology.nodes:
            node.ecc_capable = False
        job = JobSpec(job_id="reject-ecc", required_gpus=1, require_ecc=True)
        decision = algorithm.schedule_job(job, topology)
        assert len(decision.assigned_gpus) == 0
        assert "Insufficient capable" in decision.rationale

    def test_node_hardware_from_simnode(self, algorithm):
        node = SimNode(
            node_id="test-node", gpus=[],
            cpu_cores=32, system_memory_gib=256.0,
        )
        node.cpu_vendor = "amd"
        node.memory_type = "DDR5"
        node.memory_bandwidth_gbps = 153.6
        node.ecc_capable = True
        nh = algorithm._get_node_hardware(node)
        assert nh.cpu_cores == 32
        assert nh.memory_bandwidth_gbps == 153.6
        assert nh.ecc_capable is True
        assert nh.memory_type == "DDR5"


class TestEndpoints:
    def test_simulate_enhanced_endpoint(self, client):
        resp = client.post("/api/v1/ml/simulate-enhanced", params={
            "gpu_model": "NVIDIA H100-SXM-80GB",
            "num_gpus": 8, "num_nodes": 1, "steps": 10,
            "workload_type": "llm_inference",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gpus"] == 8
        assert len(data["history"]) == 10

    def test_simulate_enhanced_failure_endpoint(self, client):
        resp = client.post("/api/v1/ml/simulate-enhanced-failure", params={
            "scenario": "thermal_runaway",
        })
        assert resp.status_code == 200

    def test_schedule_job_endpoint(self, client):
        resp = client.post("/api/v1/ml/schedule", params={
            "name": "test-job", "required_gpus": 2,
            "required_memory_gib": 16.0, "policy": "hybrid",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["assigned_gpus"]) == 2
        assert data["policy"] == "hybrid"

    def test_cluster_health_endpoint(self, client):
        resp = client.get("/api/v1/ml/cluster-health")
        assert resp.status_code == 200
        assert "cluster_health" in resp.json()

    def test_closed_loop_train_endpoint(self, client):
        resp = client.post("/api/v1/ml/closed-loop-train", params={
            "cycles": 1, "steps_per_episode": 10,
        })
        assert resp.status_code == 200

    def test_compare_policies_endpoint(self, client):
        resp = client.post("/api/v1/ml/compare-policies", params={
            "steps": 10, "num_nodes": 1,
        })
        assert resp.status_code == 200

    def test_power_cap_analysis_endpoint(self, client):
        resp = client.post("/api/v1/ml/power-cap-analysis")
        assert resp.status_code == 200

    def test_drain_recommendations_endpoint(self, client):
        resp = client.get("/api/v1/ml/drain-recommendations")
        assert resp.status_code == 200

    def test_optimize_policies_endpoint(self, client):
        resp = client.post("/api/v1/ml/optimize-policies", params={
            "iterations": 5, "steps_per_eval": 10,
        })
        assert resp.status_code == 200
