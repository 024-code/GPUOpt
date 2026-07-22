from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gpuopt.domains.collectors import DomainCollector
from gpuopt.domains.stores import DomainStore, get_domain_store


@pytest.fixture(autouse=True)
def reset_store() -> None:
    get_domain_store().clear_all()


@pytest.fixture()
def seeded_store() -> DomainStore:
    store = get_domain_store()
    collector = DomainCollector(store)
    collector.seed_historical("test-cluster", minutes=3)
    return store


class TestDomainModels:
    def test_gpu_node_telemetry(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        telemetry = collector.collect_gpu_telemetry("c1", "node-a", 4)
        assert len(telemetry.gpus) == 4
        for gpu in telemetry.gpus:
            assert 0 <= gpu.utilization_gpu_pct <= 100
            assert gpu.power_watts > 0
            assert gpu.health == "healthy"

    def test_fabric_storage_telemetry(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        telemetry = collector.collect_fabric_storage_telemetry("c1")
        assert len(telemetry.nccl_events) >= 1
        assert len(telemetry.network) >= 1
        assert len(telemetry.storage) >= 1
        for ev in telemetry.nccl_events:
            assert ev.bus_bw_gbps > 0

    def test_scheduler_state(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        state = collector.collect_scheduler_state("c1")
        assert state.queue_depth >= 0
        assert state.running_jobs >= 0

    def test_training_step(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        step = collector.generate_training_step("job-1", 100, 1)
        assert step.step_time_ms > 0
        assert step.throughput_samples_per_sec > 0

    def test_inference_sample(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        sample = collector.generate_inference_sample("llama-70b")
        assert sample.ttft_ms > 0
        assert sample.tpot_ms > 0
        assert sample.model_id == "llama-70b"

    def test_tenant_quota(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        quota = collector.collect_tenant_quota("team-a")
        assert quota.tenant_id == "team-a"
        assert quota.gpu_quota > 0

    def test_cost_allocation(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        cost = collector.generate_cost_allocation("team-a")
        assert cost.total_cost_usd > 0
        assert cost.budget_usd > 0

    def test_action_chain(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        chain = collector.generate_action_chain("c1")
        assert len(chain) >= 2
        assert chain[0].action_type.value == "recommendation"
        assert chain[1].action_type.value == "approval"

    def test_training_run_summary_checkpoint_overhead(self):
        from datetime import datetime, timedelta, timezone
        summary = type("TestSummary", (), {
            "checkpoint_total_time_seconds": 1800,
            "end_time": datetime.now(timezone.utc),
            "start_time": datetime.now(timezone.utc) - timedelta(hours=10),
        })()
        # instance method requires real object
        from gpuopt.domains.models import TrainingRunSummary
        s = TrainingRunSummary(
            job_id="j1",
            start_time=datetime.now(timezone.utc) - timedelta(hours=10),
            end_time=datetime.now(timezone.utc),
            checkpoint_total_time_seconds=1800,
        )
        overhead = s.checkpoint_overhead_pct
        assert overhead > 0
        assert overhead < 100


class TestDomainStore:
    def test_store_counts(self, seeded_store: DomainStore):
        assert seeded_store.gpu_node.count() > 0
        assert seeded_store.fabric_storage.count() > 0
        assert seeded_store.scheduler_events.count() > 0
        assert seeded_store.scheduler_states.count() > 0
        assert seeded_store.training_steps.count() > 0
        assert seeded_store.inference_samples.count() > 0
        assert seeded_store.tenant_quotas.count() > 0
        assert seeded_store.cost_allocations.count() > 0
        assert seeded_store.action_events.count() > 0
        assert seeded_store.action_outcomes.count() > 0

    def test_ring_store_limit(self):
        store = get_domain_store()
        collector = DomainCollector(store)
        collector.seed_historical("c1", minutes=120)
        assert store.gpu_node.count() <= 50000
        assert store.training_steps.count() <= 100000

    def test_clear_all(self, seeded_store: DomainStore):
        seeded_store.clear_all()
        assert seeded_store.gpu_node.count() == 0

    def test_query_by_time(self, seeded_store: DomainStore):
        from datetime import datetime, timezone
        results = seeded_store.gpu_node.query(
            after=datetime(2000, 1, 1, tzinfo=timezone.utc),
            limit=5,
        )
        assert len(results) > 0


class TestDomainAPI:
    def test_collect_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/domains/collect?cluster_id=sandbox&minutes=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collected"]
        assert data["store_counts"]["gpu_node"] > 0

    def test_gpu_node_telemetry_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/gpu-node/telemetry?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert "gpus" in data[0]

    def test_gpu_node_summary_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/gpu-node/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "node_count" in data

    def test_fabric_storage_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/fabric-storage/telemetry?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) > 0

    def test_nccl_events_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/fabric-storage/nccl-events?limit=5")
        assert resp.status_code == 200

    def test_scheduler_state_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/scheduler/state?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) > 0

    def test_scheduler_events_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/scheduler/events?limit=5")
        assert resp.status_code == 200

    def test_training_steps_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/training/steps?limit=5")
        assert resp.status_code == 200

    def test_training_runs_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/training/runs?limit=5")
        assert resp.status_code == 200

    def test_inference_samples_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/inference/samples?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        if data:
            assert "ttft_ms" in data[0]

    def test_inference_summaries_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/inference/summaries?limit=5")
        assert resp.status_code == 200

    def test_tenant_quota_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/tenant/quota?limit=5")
        assert resp.status_code == 200

    def test_tenant_costs_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/tenant/costs?limit=5")
        assert resp.status_code == 200

    def test_action_events_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/actions/events?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        if data:
            assert "action_type" in data[0]

    def test_action_outcomes_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/actions/outcomes?limit=5")
        assert resp.status_code == 200

    def test_action_chain_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        events = client.get("/api/v1/domains/actions/events?limit=10").json()
        if events:
            first_id = events[0]["id"]
            chain = client.get(f"/api/v1/domains/actions/chain/{first_id}")
            assert chain.status_code == 200

    def test_counts_endpoint(self, client: TestClient):
        client.post("/api/v1/domains/collect", params={"minutes": 2})
        resp = client.get("/api/v1/domains/counts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gpu_node"] > 0
        assert data["action_events"] > 0
