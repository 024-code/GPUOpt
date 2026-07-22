from __future__ import annotations

from fastapi.testclient import TestClient

from gpuopt.federation.models import ClusterHealth, FederatedCluster, FederationRole
from gpuopt.federation.registry import FederatedClusterRegistry
from gpuopt.federation.scheduler import FederatedScheduler
from gpuopt.scheduler.rl_scheduler import RLScheduler


class TestFederatedClusterRegistry:
    def test_register_and_list(self):
        reg = FederatedClusterRegistry()
        c = reg.register("cluster-a", endpoint="https://a.example.com", region="us-east-1")
        assert c.name == "cluster-a"
        assert c.region == "us-east-1"
        assert len(reg.list()) == 1

    def test_register_duplicate_updates(self):
        reg = FederatedClusterRegistry()
        reg.register("cluster-a")
        c2 = reg.register("cluster-a", endpoint="https://new.example.com")
        assert c2.endpoint == "https://new.example.com"
        assert len(reg.list()) == 1

    def test_get_by_name(self):
        reg = FederatedClusterRegistry()
        reg.register("cluster-a")
        c = reg.get_by_name("cluster-a")
        assert c is not None
        assert c.name == "cluster-a"

    def test_unregister(self):
        reg = FederatedClusterRegistry()
        c = reg.register("cluster-a")
        assert reg.unregister(c.id) is True
        assert reg.unregister("nonexistent") is False
        assert len(reg.list()) == 0

    def test_update_health(self):
        reg = FederatedClusterRegistry()
        c = reg.register("cluster-a")
        updated = reg.update_health(c.id, ClusterHealth.ONLINE, total_gpus=16, free_gpus=8,
                                     gpu_models=["A100"], avg_utilization=50.0)
        assert updated is not None
        assert updated.health == ClusterHealth.ONLINE
        assert updated.total_gpus == 16
        assert updated.free_gpus == 8

    def test_update_health_nonexistent(self):
        reg = FederatedClusterRegistry()
        result = reg.update_health("nonexistent", ClusterHealth.ONLINE)
        assert result is None

    def test_get_state(self):
        reg = FederatedClusterRegistry()
        reg.register("cluster-a", endpoint="https://a.example.com")
        state = reg.get_state()
        assert state.total_gpus_across_clusters == 0  # not updated yet


class TestFederatedScheduler:
    def test_find_best_cluster_no_candidates(self):
        reg = FederatedClusterRegistry()
        fed = FederatedScheduler(reg)
        result = fed.find_best_cluster(required_gpus=4)
        assert result is None

    def test_find_best_cluster_selects_online(self):
        reg = FederatedClusterRegistry()
        c = reg.register("cluster-a", region="us-east-1")
        reg.update_health(c.id, ClusterHealth.ONLINE, total_gpus=16, free_gpus=8)

        fed = FederatedScheduler(reg)
        best = fed.find_best_cluster(required_gpus=4)
        assert best is not None
        assert best.name == "cluster-a"

    def test_skips_offline_and_draining(self):
        reg = FederatedClusterRegistry()
        online = reg.register("online-cluster")
        reg.update_health(online.id, ClusterHealth.ONLINE, total_gpus=8, free_gpus=4)
        offline = reg.register("offline-cluster")
        reg.update_health(offline.id, ClusterHealth.OFFLINE, total_gpus=8, free_gpus=4)
        draining = reg.register("draining-cluster")
        reg.update_health(draining.id, ClusterHealth.ONLINE, total_gpus=8, free_gpus=4)
        reg.get(draining.id).role = FederationRole.DRAINING

        fed = FederatedScheduler(reg)
        best = fed.find_best_cluster(required_gpus=2)
        assert best is not None
        assert best.name == "online-cluster"

    def test_schedule_across_clusters(self):
        reg = FederatedClusterRegistry()
        c = reg.register("target-cluster", region="us-east-1")
        reg.update_health(c.id, ClusterHealth.ONLINE, total_gpus=16, free_gpus=8)

        fed = FederatedScheduler(reg, RLScheduler())
        result = fed.schedule_across_clusters(required_gpus=2, region="us-east-1")
        assert result["mode"] == "federated"
        assert result["status"] in ("scheduled", "queued")
        assert result["cluster"] == "target-cluster"

    def test_schedule_no_cluster_queues(self):
        reg = FederatedClusterRegistry()
        fed = FederatedScheduler(reg, RLScheduler())
        result = fed.schedule_across_clusters(required_gpus=100)
        assert result["status"] == "queued"

    def test_list_and_get_workloads(self):
        reg = FederatedClusterRegistry()
        fed = FederatedScheduler(reg)
        assert len(fed.list_workloads()) == 0
        c = reg.register("c1")
        reg.update_health(c.id, ClusterHealth.ONLINE, total_gpus=8, free_gpus=4)
        fed.schedule_across_clusters(required_gpus=2, workload_name="test-wl")
        assert len(fed.list_workloads()) > 0

    def test_get_state(self):
        reg = FederatedClusterRegistry()
        c = reg.register("cluster-a")
        reg.update_health(c.id, ClusterHealth.ONLINE, total_gpus=8, free_gpus=4)
        fed = FederatedScheduler(reg)
        state = fed.get_state()
        assert state["total_clusters"] == 1
        assert state["total_gpus"] == 8


class TestFederationAPI:
    def test_register_cluster_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/federation/clusters", json={
            "name": "api-cluster", "endpoint": "https://cluster.example.com",
            "region": "us-west-2", "environment": "production",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "registered"
        assert data["cluster"]["name"] == "api-cluster"

    def test_list_clusters_endpoint(self, client: TestClient):
        client.post("/api/v1/federation/clusters", json={"name": "list-cluster"})
        resp = client.get("/api/v1/federation/clusters")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_get_cluster_endpoint(self, client: TestClient):
        created = client.post("/api/v1/federation/clusters", json={"name": "get-cluster"}).json()
        cid = created["cluster"]["id"]
        resp = client.get(f"/api/v1/federation/clusters/{cid}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get-cluster"

    def test_delete_cluster_endpoint(self, client: TestClient):
        created = client.post("/api/v1/federation/clusters", json={"name": "delete-cluster"}).json()
        cid = created["cluster"]["id"]
        resp = client.delete(f"/api/v1/federation/clusters/{cid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/v1/federation/clusters/{cid}")
        assert resp.status_code == 404

    def test_update_health_endpoint(self, client: TestClient):
        created = client.post("/api/v1/federation/clusters", json={"name": "health-cluster"}).json()
        cid = created["cluster"]["id"]
        resp = client.put(f"/api/v1/federation/clusters/{cid}/health", json={
            "health": "online", "total_gpus": 16, "free_gpus": 8,
            "gpu_models": ["H100", "A100"], "avg_utilization": 45.0,
        })
        assert resp.status_code == 200
        assert resp.json()["cluster"]["total_gpus"] == 16

    def test_federated_schedule_endpoint(self, client: TestClient):
        created = client.post("/api/v1/federation/clusters", json={"name": "sched-cluster", "region": "us-east-1"}).json()
        cid = created["cluster"]["id"]
        client.put(f"/api/v1/federation/clusters/{cid}/health", json={
            "health": "online", "total_gpus": 16, "free_gpus": 8,
        })
        resp = client.post("/api/v1/federation/schedule", json={
            "required_gpus": 2, "region": "us-east-1",
        })
        assert resp.status_code == 200
        assert resp.json()["mode"] == "federated"

    def test_federation_state_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/federation/state")
        assert resp.status_code == 200
        assert "total_clusters" in resp.json()
