from __future__ import annotations

import threading
import time

import pytest

from gpuopt.autoscaler import AutoscalerEngine, AutoscalingConfig, NodeGroupConfig, ScalingPolicy
from gpuopt.gpu_monitor import GPUMonitor, GPUProcessInfo
from gpuopt.preemption import PreemptionEngine, PreemptionPolicy, PreemptionPolicyConfig, PriorityClass


class TestGPUMonitor:
    def test_collect_returns_snapshot(self):
        monitor = GPUMonitor()
        snap = monitor.collect()
        assert snap.total_gpus >= 0
        assert snap.total_memory_mb >= 0
        assert snap.used_memory_mb >= 0
        assert snap.free_memory_mb >= 0
        assert len(snap.devices) == snap.total_gpus

    def test_snapshot_device_fields(self):
        monitor = GPUMonitor()
        snap = monitor.collect()
        for dev in snap.devices:
            assert dev.index >= 0
            assert dev.model
            assert dev.memory_total_mb > 0
            assert dev.memory_used_mb >= 0
            assert dev.memory_free_mb >= 0
            assert 0 <= dev.utilization_gpu_percent <= 100
            assert dev.temperature_celsius >= 0

    def test_collect_has_processes(self):
        monitor = GPUMonitor()
        snap = monitor.collect()
        for dev in snap.devices:
            for proc in dev.processes:
                assert isinstance(proc.pid, int)
                assert proc.used_gpu_memory_mb >= 0

    def test_mock_snapshot_consistent(self):
        monitor = GPUMonitor()
        snap = monitor._mock_snapshot()
        assert snap.total_gpus == len(snap.devices)
        total = sum(d.memory_total_mb for d in snap.devices)
        used = sum(d.memory_used_mb for d in snap.devices)
        free = sum(d.memory_free_mb for d in snap.devices)
        assert snap.total_memory_mb == total
        assert snap.used_memory_mb == used
        assert snap.free_memory_mb == free
        assert snap.total_memory_mb == snap.used_memory_mb + snap.free_memory_mb

    def test_start_stop_threaded(self):
        monitor = GPUMonitor(poll_interval=0.2)
        try:
            monitor.start()
            assert monitor._running
            assert monitor._thread is not None
            assert monitor._thread.is_alive()
            time.sleep(0.5)
            snap = monitor.get_snapshot()
            assert snap is not None
            assert snap.total_gpus > 0
        finally:
            monitor.stop()
        assert not monitor._running
        assert monitor._thread is None

    def test_get_snapshot_none_before_first_poll(self):
        monitor = GPUMonitor(poll_interval=60)
        assert monitor.get_snapshot() is None


class TestPreemptionEngine:
    def test_never_policy_produces_no_actions(self):
        config = PreemptionPolicyConfig(policy=PreemptionPolicy.NEVER)
        engine = PreemptionEngine(config=config)
        actions = engine.cycle()
        assert actions == []

    def test_cycle_returns_preemption_actions(self):
        engine = PreemptionEngine()
        actions = engine.cycle()
        for action in actions:
            assert action.workload_name
            assert action.namespace
            assert action.priority in PriorityClass
            assert action.reason
            assert action.eviction_strategy
            assert action.status in ("executed", "pending", "failed")
            assert action.initiated_at

    def test_start_stop(self):
        engine = PreemptionEngine()
        engine.start()
        assert engine._running
        engine.stop()
        assert not engine._running

    def test_get_history(self):
        engine = PreemptionEngine()
        assert engine.get_history() == []
        engine.cycle()
        history = engine.get_history()
        assert len(history) >= 0

    def test_config_update(self):
        engine = PreemptionEngine()
        original = engine.config
        new_config = PreemptionPolicyConfig(
            policy=PreemptionPolicy.NEVER,
            min_priority_delta=200,
            preempt_oldest_first=False,
            max_preemptions_per_cycle=10,
        )
        engine.config = new_config
        assert engine.config.policy == PreemptionPolicy.NEVER
        assert engine.config.min_priority_delta == 200
        assert engine.config.max_preemptions_per_cycle == 10
        engine.config = original

    def test_preemption_priority_sorting(self):
        high = {"workload_name": "critical-job", "namespace": "default", "priority": PriorityClass.CRITICAL, "gpu_claimed": 8, "age_minutes": 5}
        low = {"workload_name": "batch-job", "namespace": "default", "priority": PriorityClass.BATCH, "gpu_claimed": 4, "age_minutes": 60}
        engine = PreemptionEngine(config=PreemptionPolicyConfig(min_priority_delta=50))
        collisions = engine._detect_collisions([high, low])
        assert len(collisions) == 1
        assert collisions[0][0]["workload_name"] == "critical-job"
        assert collisions[0][1]["workload_name"] == "batch-job"

    def test_same_priority_no_collisions(self):
        wl1 = {"workload_name": "job-a", "namespace": "default", "priority": PriorityClass.PRODUCTION, "gpu_claimed": 4, "age_minutes": 10}
        wl2 = {"workload_name": "job-b", "namespace": "default", "priority": PriorityClass.PRODUCTION, "gpu_claimed": 4, "age_minutes": 20}
        engine = PreemptionEngine()
        collisions = engine._detect_collisions([wl1, wl2])
        assert len(collisions) == 0

    def test_history_smoke(self):
        engine = PreemptionEngine()
        actions = engine.cycle()
        history = engine.get_history(limit=5)
        assert len(history) <= 5 + len(actions)
        for action in history:
            assert action.status in ("executed", "pending", "failed")

    def test_config_set_and_get(self):
        config = PreemptionPolicyConfig(policy=PreemptionPolicy.PREEMPT_LOWER_PRIORITY, min_priority_delta=100)
        engine = PreemptionEngine(config=config)
        assert engine.config.policy == PreemptionPolicy.PREEMPT_LOWER_PRIORITY
        assert engine.config.min_priority_delta == 100


class TestAutoscalerEngine:
    def test_initial_status(self):
        engine = AutoscalerEngine()
        status = engine.get_status()
        assert not status.running
        assert status.event_count == 0
        assert status.last_event is None

    def test_manual_scale_up(self):
        engine = AutoscalerEngine()
        event = engine.scale_manual("test-group", 5)
        assert event.direction.value == "up"
        assert event.node_group == "test-group"
        assert event.target_size == 5
        assert event.status in ("executed", "failed")
        assert event.reason

    def test_manual_scale_down(self):
        engine = AutoscalerEngine()
        event = engine.scale_manual("test-group", 1)
        assert event.direction.value in ("up", "down")
        assert event.node_group == "test-group"

    def test_get_events(self):
        engine = AutoscalerEngine()
        assert engine.get_events() == []
        engine.scale_manual("test-group", 3)
        assert len(engine.get_events()) == 1
        engine.scale_manual("test-group", 5)
        assert len(engine.get_events()) == 2

    def test_start_stop(self):
        engine = AutoscalerEngine()
        engine.start()
        assert engine._running
        engine.stop()
        assert not engine._running

    def test_config_defaults(self):
        config = engine = AutoscalerEngine().config
        assert config.policy == ScalingPolicy.AUTOMATIC
        assert config.scale_up_threshold == 80.0
        assert config.scale_down_threshold == 30.0
        assert config.cooldown_seconds == 300
        assert config.min_nodes == 1
        assert config.max_nodes == 20

    def test_config_update(self):
        engine = AutoscalerEngine()
        original = engine.config
        new_config = AutoscalingConfig(
            policy=ScalingPolicy.MANUAL,
            scale_up_threshold=90.0,
            scale_down_threshold=20.0,
            cooldown_seconds=600,
            min_nodes=2,
            max_nodes=15,
            node_groups=[NodeGroupConfig(name="gpu-group", min_size=2, max_size=10, current_size=3, gpu_type="A100", gpus_per_node=8)],
        )
        engine.config = new_config
        assert engine.config.policy == ScalingPolicy.MANUAL
        assert engine.config.scale_up_threshold == 90.0
        assert engine.config.scale_down_threshold == 20.0
        assert engine.config.cooldown_seconds == 600
        assert engine.config.min_nodes == 2
        assert engine.config.max_nodes == 15
        assert len(engine.config.node_groups) == 1
        assert engine.config.node_groups[0].name == "gpu-group"
        engine.config = original

    def test_events_bound(self):
        engine = AutoscalerEngine()
        for _ in range(10):
            engine.scale_manual("test-group", 3)
        assert len(engine.get_events(limit=3)) == 3

    def test_events_ordered(self):
        engine = AutoscalerEngine()
        for target in [2, 4, 6]:
            engine.scale_manual("test-group", target)
        events = engine.get_events()
        assert len(events) == 3
        assert events[-1].target_size == 6


class TestAutoscalerConfig:
    def test_node_group_config_defaults(self):
        ng = NodeGroupConfig(name="test")
        assert ng.name == "test"
        assert ng.min_size == 0
        assert ng.max_size == 10
        assert ng.current_size == 1
        assert ng.gpu_type == ""
        assert ng.gpus_per_node == 8
        assert ng.labels == {}
        assert ng.taints == []

    def test_autoscaling_config_defaults(self):
        config = AutoscalingConfig()
        assert config.policy == ScalingPolicy.AUTOMATIC
        assert config.scale_up_threshold == 80.0
        assert config.scale_down_threshold == 30.0
        assert config.scale_up_increment == 1
        assert config.scale_down_decrement == 1
        assert config.cooldown_seconds == 300
        assert config.min_nodes == 1
        assert config.max_nodes == 20
        assert config.node_groups == []


@pytest.fixture
def client(tmp_path):
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = str(tmp_path / "test_monitoring.db")
    from gpuopt.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    from gpuopt.config import get_settings
    get_settings.cache_clear()


class TestMonitoringAPI:
    def test_get_gpu_snapshot(self, client):
        resp = client.get("/api/v1/monitoring/gpu/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_gpus" in data
        assert "total_memory_mb" in data
        assert "devices" in data
        for dev in data["devices"]:
            assert "index" in dev
            assert "model" in dev
            assert "memory_total_mb" in dev
            assert "processes" in dev

    def test_get_gpu_snapshot_cached_not_started(self, client):
        resp = client.get("/api/v1/monitoring/gpu/snapshot/cached")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_data"

    def test_start_stop_gpu_monitor(self, client):
        resp = client.post("/api/v1/monitoring/gpu/start", json={"poll_interval": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        time.sleep(1)
        resp2 = client.get("/api/v1/monitoring/gpu/status")
        assert resp2.status_code == 200
        status_data = resp2.json()
        assert status_data["running"]
        resp3 = client.post("/api/v1/monitoring/gpu/stop")
        assert resp3.status_code == 200
        assert resp3.json()["status"] == "stopped"

    def test_preemption_cycle(self, client):
        resp = client.post("/api/v1/monitoring/preemption/cycle")
        assert resp.status_code == 200
        actions = resp.json()
        assert isinstance(actions, list)
        for action in actions:
            assert "workload_name" in action
            assert "reason" in action
            assert "status" in action

    def test_preemption_config(self, client):
        resp = client.get("/api/v1/monitoring/preemption/config")
        assert resp.status_code == 200
        config = resp.json()
        assert "policy" in config
        assert "min_priority_delta" in config
        assert "max_preemptions_per_cycle" in config

    def test_update_preemption_config(self, client):
        resp = client.put("/api/v1/monitoring/preemption/config", json={"policy": "Never", "min_priority_delta": 200})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
        assert data["config"]["policy"] == "Never"
        assert data["config"]["min_priority_delta"] == 200

    def test_apply_preemption_policy(self, client):
        resp = client.post("/api/v1/monitoring/preemption/apply", json={
            "workload_name": "test-job",
            "namespace": "default",
            "priority": "production",
            "policy": "PreemptLowerPriority",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("applied", "failed")

    def test_preemption_start_stop(self, client):
        resp = client.post("/api/v1/monitoring/preemption/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        resp = client.post("/api/v1/monitoring/preemption/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_preemption_history(self, client):
        client.post("/api/v1/monitoring/preemption/cycle")
        resp = client.get("/api/v1/monitoring/preemption/history")
        assert resp.status_code == 200
        history = resp.json()
        assert isinstance(history, list)

    def test_autoscaler_status(self, client):
        resp = client.get("/api/v1/monitoring/autoscaler/status")
        assert resp.status_code == 200
        status = resp.json()
        assert "running" in status
        assert "event_count" in status

    def test_autoscaler_events(self, client):
        client.post("/api/v1/monitoring/autoscaler/scale", json={"node_group": "test", "target_size": 4})
        resp = client.get("/api/v1/monitoring/autoscaler/events")
        assert resp.status_code == 200
        events = resp.json()
        assert isinstance(events, list)

    def test_autoscaler_scale_manual(self, client):
        resp = client.post("/api/v1/monitoring/autoscaler/scale", json={"node_group": "gpu-group", "target_size": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "scaled"
        assert data["event"]["target_size"] == 5
        assert data["event"]["node_group"] == "gpu-group"

    def test_autoscaler_start_stop(self, client):
        resp = client.post("/api/v1/monitoring/autoscaler/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        resp = client.post("/api/v1/monitoring/autoscaler/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_autoscaler_config_get(self, client):
        resp = client.get("/api/v1/monitoring/autoscaler/config")
        assert resp.status_code == 200
        config = resp.json()
        assert config["policy"] == "automatic"
        assert config["scale_up_threshold"] == 80.0
        assert config["min_nodes"] == 1

    def test_autoscaler_config_update(self, client):
        resp = client.put("/api/v1/monitoring/autoscaler/config", json={
            "policy": "manual",
            "scale_up_threshold": 90,
            "min_nodes": 2,
            "max_nodes": 10,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"
        resp2 = client.get("/api/v1/monitoring/autoscaler/config")
        config = resp2.json()
        assert config["policy"] == "manual"
        assert config["scale_up_threshold"] == 90.0
        assert config["min_nodes"] == 2

    def test_autoscaler_events_bound(self, client):
        for i in range(5):
            client.post("/api/v1/monitoring/autoscaler/scale", json={"node_group": "g", "target_size": i + 1})
        resp = client.get("/api/v1/monitoring/autoscaler/events?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2
