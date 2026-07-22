from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


from gpuopt.schemas import (
    SlurmJobControlRequest,
    SlurmJobControlResult,
    SlurmReservation,
    SlurmReservationRequest,
)
from gpuopt.slurm_actions import SlurmActionAdapter

MOCK_SLURM_PATH = "sandbox/mock-clusters/mock-slurm.json"


@pytest.fixture()
def slurm_cluster(client: TestClient) -> str:
    resp = client.post("/api/v1/clusters", json={
        "name": "slurm-test",
        "environment": "test",
        "connector_type": "slurm",
        "options": {"mock_slurm_data": MOCK_SLURM_PATH},
    })
    return resp.json()["id"]


@pytest.fixture(autouse=True)
def reset_mock_state() -> None:
    SlurmActionAdapter.reset_mock_state()

@pytest.fixture()
def adapter(client: TestClient) -> SlurmActionAdapter:
    from gpuopt.dependencies import get_repository
    repo = get_repository()
    return SlurmActionAdapter(repo)


class TestSlurmJobControl:
    def test_submit_job(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        req = SlurmJobControlRequest(
            action="submit", job_name="test_job", partition="gpu",
            gpu_count=8, cpu_count=32, time_limit_minutes=120,
        )
        result = adapter.submit_job(slurm_cluster, req)
        assert result.success
        assert result.action == "submit"
        assert result.job_id > 0

    def test_cancel_job(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        req = SlurmJobControlRequest(action="submit", job_name="cancel_test")
        submit = adapter.submit_job(slurm_cluster, req)
        result = adapter.cancel_job(slurm_cluster, submit.job_id)
        assert result.success
        assert result.action == "cancel"

    def test_cancel_nonexistent_job(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        result = adapter.cancel_job(slurm_cluster, 99999)
        assert not result.success

    def test_hold_and_release_job(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        req = SlurmJobControlRequest(action="submit", job_name="hold_test")
        submit = adapter.submit_job(slurm_cluster, req)
        hold = adapter.hold_job(slurm_cluster, submit.job_id)
        assert hold.success
        release = adapter.release_job(slurm_cluster, submit.job_id)
        assert release.success

    def test_modify_job(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        req = SlurmJobControlRequest(action="submit", job_name="mod_test")
        submit = adapter.submit_job(slurm_cluster, req)
        mod = adapter.modify_job(slurm_cluster, SlurmJobControlRequest(
            action="modify", job_id=submit.job_id, time_limit_minutes=240,
        ))
        assert mod.success
        assert mod.action == "modify"


class TestSlurmReservations:
    def test_create_reservation(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        req = SlurmReservationRequest(
            name="test-reservation", partition="gpu",
            node_count=2, duration_minutes=120,
        )
        reservation = adapter.create_reservation(slurm_cluster, req)
        assert reservation.name == "test-reservation"
        assert reservation.state == "active"
        assert reservation.duration_minutes == 120

    def test_list_reservations(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        req = SlurmReservationRequest(
            name="list-test", partition="gpu", duration_minutes=60,
        )
        adapter.create_reservation(slurm_cluster, req)
        reservations = adapter.list_reservations(slurm_cluster)
        assert len(reservations) >= 1
        names = [r.name for r in reservations]
        assert "list-test" in names

    def test_delete_reservation(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        req = SlurmReservationRequest(
            name="delete-test", partition="gpu", duration_minutes=30,
        )
        adapter.create_reservation(slurm_cluster, req)
        result = adapter.delete_reservation(slurm_cluster, "delete-test")
        assert result
        reservations = adapter.list_reservations(slurm_cluster)
        names = [r.name for r in reservations]
        assert "delete-test" not in names

    def test_delete_nonexistent_reservation(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        result = adapter.delete_reservation(slurm_cluster, "nonexistent")
        assert not result


class TestSlurmAccounting:
    def test_get_accounting_existing(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        result = adapter.get_job_accounting(slurm_cluster, 999)
        assert result["job_id"] == 999
        assert len(result["entries"]) > 0
        assert result["entries"][0]["job_id"] == "999"

    def test_get_accounting_nonexistent(self, slurm_cluster: str, adapter: SlurmActionAdapter):
        result = adapter.get_job_accounting(slurm_cluster, 99999)
        assert result["job_id"] == 99999
        assert len(result["entries"]) == 0


class TestSlurmAPI:
    def test_submit_job_api(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "slurm-api-test",
            "environment": "test",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": MOCK_SLURM_PATH},
        }).json()["id"]
        resp = client.post(f"/api/v1/slurm/{cid}/jobs/submit", json={
            "action": "submit", "job_name": "api_job",
            "partition": "gpu", "gpu_count": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert data["job_id"] > 0

    def test_cancel_job_api(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "slurm-cancel-test",
            "environment": "test",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": MOCK_SLURM_PATH},
        }).json()["id"]
        submit = client.post(f"/api/v1/slurm/{cid}/jobs/submit", json={
            "action": "submit", "job_name": "cancel_me",
        })
        jid = submit.json()["job_id"]
        resp = client.post(f"/api/v1/slurm/{cid}/jobs/{jid}/cancel")
        assert resp.status_code == 200
        assert resp.json()["success"]

    def test_create_reservation_api(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "slurm-res-test",
            "environment": "test",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": MOCK_SLURM_PATH},
        }).json()["id"]
        resp = client.post(f"/api/v1/slurm/{cid}/reservations", json={
            "name": "api-reservation", "partition": "gpu",
            "node_count": 1, "duration_minutes": 60,
        })
        assert resp.status_code == 200
        assert resp.json()["state"] == "active"

    def test_list_reservations_api(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "slurm-list-res-test",
            "environment": "test",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": MOCK_SLURM_PATH},
        }).json()["id"]
        client.post(f"/api/v1/slurm/{cid}/reservations", json={
            "name": "list-res-1", "partition": "gpu", "duration_minutes": 60,
        })
        resp = client.get(f"/api/v1/slurm/{cid}/reservations")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_accounting_api(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "slurm-acct-test",
            "environment": "test",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": MOCK_SLURM_PATH},
        }).json()["id"]
        resp = client.get(f"/api/v1/slurm/{cid}/accounting/999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == 999


class TestSlurmCliParsing:
    def test_parse_slurm_time_standard(self):
        from gpuopt.connectors.slurm import SlurmConnector

        class MockCluster:
            id = "test"
            name = "test"
            options = {}
            in_cluster = False
            kubeconfig_path = None
            kube_context = None

        conn = SlurmConnector(MockCluster())
        assert conn._parse_slurm_time("1:30") == 90
        assert conn._parse_slurm_time("2:00:00") == 120
        assert conn._parse_slurm_time("UNLIMITED") == 0
        assert conn._parse_slurm_time("") == 0
        assert conn._parse_slurm_time("NOT_SET") == 0

    def test_parse_slurm_time_days(self):
        from gpuopt.connectors.slurm import SlurmConnector

        class MockCluster:
            id = "test"
            name = "test"
            options = {}
            in_cluster = False
            kubeconfig_path = None
            kube_context = None

        conn = SlurmConnector(MockCluster())
        result = conn._parse_slurm_time("1-00:00:00")
        assert result == 1440  # 1 day in minutes

    def test_parse_sinfo_nodes_format_basic(self):
        from gpuopt.connectors.slurm import SlurmConnector

        class MockCluster:
            id = "test"
            name = "test"
            options = {}
            in_cluster = False
            kubeconfig_path = None
            kube_context = None

        conn = SlurmConnector(MockCluster())
        output = "node-1|idle|64|256000|gpu:a100:8|nvlink"
        nodes = conn._parse_sinfo_nodes_format(output)
        assert len(nodes) == 1
        assert nodes[0].node_name == "node-1"
        assert nodes[0].state == "idle"
        assert nodes[0].cpu_count == 64
        assert nodes[0].gpu_count == 8
        assert nodes[0].gpu_model == "a100"

    def test_parse_sinfo_nodes_multiple(self):
        from gpuopt.connectors.slurm import SlurmConnector

        class MockCluster:
            id = "test"
            name = "test"
            options = {}
            in_cluster = False
            kubeconfig_path = None
            kube_context = None

        conn = SlurmConnector(MockCluster())
        output = "node-1|idle|64|256000|gpu:a100:4|nvlink\nnode-2|mix|32|128000|gpu:v100:2|ib"
        nodes = conn._parse_sinfo_nodes_format(output)
        assert len(nodes) == 2
        assert nodes[0].gpu_count == 4
        assert nodes[1].gpu_count == 2

    def test_parse_jobs_from_delim(self):
        from gpuopt.connectors.slurm import SlurmConnector

        class MockCluster:
            id = "test"
            name = "test"
            options = {}
            in_cluster = False
            kubeconfig_path = None
            kube_context = None

        conn = SlurmConnector(MockCluster())
        output = "12345|train_job|gpu|alice|RUNNING|1|8|16G|1:00:00|0:30:00|node-1\n12346|data_job|cpu|bob|PENDING|2|4|8G|2:00:00|0:00:00|"
        pending, running = conn._parse_jobs_from_delim(output)
        assert len(running) == 1
        assert len(pending) == 1
        assert running[0].job_id == 12345
        assert running[0].job_name == "train_job"
        assert pending[0].job_id == 12346
        assert pending[0].state == "PENDING"

    def test_parse_jobs_from_delim_empty(self):
        from gpuopt.connectors.slurm import SlurmConnector

        class MockCluster:
            id = "test"
            name = "test"
            options = {}
            in_cluster = False
            kubeconfig_path = None
            kube_context = None

        conn = SlurmConnector(MockCluster())
        pending, running = conn._parse_jobs_from_delim("")
        assert len(pending) == 0
        assert len(running) == 0

    def test_parse_jobs_memory_units(self):
        from gpuopt.connectors.slurm import SlurmConnector

        class MockCluster:
            id = "test"
            name = "test"
            options = {}
            in_cluster = False
            kubeconfig_path = None
            kube_context = None

        conn = SlurmConnector(MockCluster())
        output = "1|j1|p|u|RUNNING|1|4|8G|10:00|1:00|n1\n2|j2|p|u|RUNNING|2|8|16384M|20:00|2:00|n2"
        pending, running = conn._parse_jobs_from_delim(output)
        assert running[0].memory_mb == 8192  # 8G
        assert running[1].memory_mb == 16384  # 16384M
