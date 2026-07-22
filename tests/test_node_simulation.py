from __future__ import annotations

import pytest

from gpuopt.ml.node_simulation import (
    ClusterTopology,
    CoolingSystem,
    CoolingType,
    EnhancedSimulationEngine,
    GpuDegradation,
    GpuSpec,
    PowerDistributionUnit,
    SimGpu,
    SimNode,
)


class TestClusterTopology:
    def test_build_dgx_h100(self):
        topo = ClusterTopology().build_dgx_h100(2)
        assert topo.total_gpus == 16
        assert len(topo.nodes) == 2
        assert topo.nodes[0].cooling.cooling_type == CoolingType.DIRECT_LIQUID

    def test_build_rtx_cluster(self):
        topo = ClusterTopology().build_rtx_cluster(16)
        assert topo.total_gpus == 16
        assert len(topo.nodes) == 1
        assert topo.nodes[0].cooling.cooling_type == CoolingType.AIR

    def test_get_gpu(self):
        topo = ClusterTopology().build_dgx_h100(1)
        gpu = topo.get_gpu(0, 3)
        assert gpu.index == 3
        assert gpu.node_id == "dgx-h100-00"

    def test_pdu_headroom(self):
        pdu = PowerDistributionUnit(max_power_kw=10.0)
        assert pdu.headroom_kw == 10.0
        pdu.assign_load(3.0)
        assert pdu.headroom_kw < 10.0

    def test_pdu_can_serve(self):
        pdu = PowerDistributionUnit(max_power_kw=5.0, circuits=1,
                                     circuit_capacity_kw=5.0)
        assert pdu.can_serve(2.0)
        pdu.assign_load(4.0)
        assert not pdu.can_serve(2.0)


class TestGpuDegradation:
    def test_wear_factor_increases_with_use(self):
        d = GpuDegradation(total_hours_operational=50000.0)
        assert d.wear_factor > 0.01

    def test_wear_factor_capped(self):
        d = GpuDegradation(total_hours_operational=1e6,
                           thermal_cycles_above_80c=100000,
                           accumulated_thermal_stress=50000)
        assert d.wear_factor <= 1.0

    def test_mtbf_multiplier(self):
        d = GpuDegradation(total_hours_operational=25000.0)
        assert d.mtbf_multiplier < 1.0


class TestSimGpu:
    def test_memory_pct(self):
        gpu = SimGpu(index=0, node_id="n1",
                      spec=GpuSpec(memory_gib=80.0),
                      memory_used_gib=40.0)
        assert gpu.memory_pct == 50.0

    def test_memory_pct_zero_division(self):
        gpu = SimGpu(index=0, node_id="n1",
                      spec=GpuSpec(memory_gib=0))
        assert gpu.memory_pct == 0.0


class TestSimNode:
    def test_total_gpu_power(self):
        gpus = [
            SimGpu(0, "n1", GpuSpec(), power_draw_watts=300.0),
            SimGpu(1, "n1", GpuSpec(), power_draw_watts=200.0),
        ]
        node = SimNode("n1", gpus)
        assert node.total_gpu_power_watts == 500.0

    def test_total_power_includes_overhead(self):
        gpu = SimGpu(0, "n1", GpuSpec(tdp_watts=400.0), power_draw_watts=300.0)
        node = SimNode("n1", [gpu], cooling=CoolingSystem(pue=1.5))
        total = node.total_power_watts
        assert total > 300.0


class TestEnhancedSimulationEngine:
    def test_init_topology(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_dgx_h100(1)
        engine.init_topology(topo)
        assert engine.topology is not None
        assert engine.step_count == 0

    def test_step_simulation(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_dgx_h100(1)
        engine.init_topology(topo)

        engine.step_simulation(profile={
            "gpu_util_target": 60.0, "memory_target_pct": 50.0,
            "tensor_intensity": 0.6, "mem_intensity": 0.5,
        })
        assert engine.step_count == 1
        assert engine.time_hours > 0

    def test_simulate_returns_history(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_dgx_h100(1)
        result = engine.simulate(topo, steps=10)
        assert len(result["history"]) == 10
        assert result["total_gpus"] == 8

    def test_simulate_rtx_cluster(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_rtx_cluster(4)
        result = engine.simulate(topo, steps=5)
        assert result["total_gpus"] == 4

    def test_capture_snapshot(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_dgx_h100(1)
        engine.init_topology(topo)
        snap = engine.capture_snapshot()
        assert "nodes" in snap
        assert "aggregate" in snap
        assert snap["aggregate"]["total_gpus"] == 8

    def test_simulate_failure_thermal(self):
        engine = EnhancedSimulationEngine()
        result = engine.simulate_failure_scenario("thermal_runaway")
        assert "scenario" in result
        assert result["scenario"] == "thermal_runaway"

    def test_simulate_failure_memory_leak(self):
        engine = EnhancedSimulationEngine()
        result = engine.simulate_failure_scenario("memory_leak")
        assert result["scenario"] == "memory_leak"

    def test_simulate_failure_xid_storm(self):
        engine = EnhancedSimulationEngine()
        result = engine.simulate_failure_scenario("xid_storm")
        assert result["scenario"] == "xid_storm"

    def test_simulate_failure_power_surge(self):
        engine = EnhancedSimulationEngine()
        result = engine.simulate_failure_scenario("power_surge")
        assert result["scenario"] == "power_surge"

    def test_simulate_failure_fan_failure(self):
        engine = EnhancedSimulationEngine()
        result = engine.simulate_failure_scenario("fan_failure")
        assert result["scenario"] == "fan_failure"

    def test_error_injection(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_dgx_h100(1)
        engine.init_topology(topo)
        high_stress_gpu = topo.get_gpu(0, 0)
        high_stress_gpu.gpu_temp_c = 95.0
        high_stress_gpu.engine_util_pct = 95.0
        engine.inject_errors(high_stress_gpu, stress=2.0, profile_intensity=1.0)
        assert isinstance(high_stress_gpu.xid_errors, int)
        assert isinstance(high_stress_gpu.ecc_corrected, int)

    def test_degradation_accumulates(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_dgx_h100(1)
        engine.init_topology(topo)
        gpu = topo.get_gpu(0, 0)
        gpu.gpu_temp_c = 82.0
        gpu.engine_util_pct = 90.0
        for _ in range(10):
            engine.update_degradation(gpu, dt_hours=1.0)
        assert gpu.degradation.total_hours_operational >= 10.0
        assert gpu.degradation.thermal_cycles_above_80c >= 10
        assert gpu.degradation.accumulated_thermal_stress > 0

    def test_event_log(self):
        engine = EnhancedSimulationEngine()
        topo = ClusterTopology().build_dgx_h100(1)
        engine.init_topology(topo)
        engine.log_event("test", "n1", 0, "test event")
        assert len(engine.event_log) == 1
        assert engine.event_log[0]["event_type"] == "test"
