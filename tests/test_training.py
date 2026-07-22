from __future__ import annotations

from uuid import uuid4

from gpuopt.schemas import (
    HPOConfig,
    TrainingFramework,
    TrainingJobStatus,
)
from gpuopt.training import TrainingService


def test_register_and_get_job():
    svc = TrainingService()
    cid = uuid4()
    job = svc.register_job(
        cluster_id=cid,
        job_name="test-train",
        framework=TrainingFramework.PYTORCH,
        gpu_count=8,
        batch_size=64,
        precision="bf16",
    )
    assert job.job_name == "test-train"
    assert job.framework == TrainingFramework.PYTORCH
    assert job.gpu_count == 8
    assert job.status == TrainingJobStatus.PENDING

    fetched = svc.get_job(job.id)
    assert fetched is not None
    assert fetched.id == job.id


def test_register_and_get_job2():
    svc = TrainingService()
    cid = uuid4()
    job = svc.register_job(
        cluster_id=cid,
        job_name="test-train2",
        framework=TrainingFramework.PYTORCH,
        gpu_count=8,
        batch_size=64,
        precision="bf16",
    )
    fetched = svc.get_job(job.id)
    assert fetched is not None
    assert fetched.job_name == "test-train2"


def test_list_jobs():
    svc = TrainingService()
    cid = uuid4()
    svc.register_job(cluster_id=cid, job_name="job-a")
    svc.register_job(cluster_id=cid, job_name="job-b")
    other = uuid4()
    svc.register_job(cluster_id=other, job_name="job-c")
    jobs = svc.list_jobs(cid)
    assert len(jobs) == 2
    all_jobs = svc.list_jobs()
    assert len(all_jobs) >= 3


def test_update_job():
    svc = TrainingService()
    cid = uuid4()
    job = svc.register_job(cluster_id=cid, job_name="update-test")
    updated = svc.update_job(
        job.id,
        status=TrainingJobStatus.RUNNING,
        loss_value=0.05,
        epochs_completed=3,
        avg_gpu_utilization=78.5,
        throughput_samples_per_sec=120.0,
    )
    assert updated is not None
    assert updated.status == TrainingJobStatus.RUNNING
    assert updated.loss_value == 0.05
    assert updated.epochs_completed == 3
    assert updated.avg_gpu_utilization == 78.5
    assert updated.throughput_samples_per_sec == 120.0


def test_delete_job():
    svc = TrainingService()
    job = svc.register_job(cluster_id=uuid4(), job_name="delete-me")
    assert svc.delete_job(job.id) is True
    assert svc.delete_job(job.id) is False
    assert svc.get_job(job.id) is None


def test_profile_job():
    svc = TrainingService()
    cid = uuid4()
    job = svc.register_job(
        cluster_id=cid,
        job_name="profile-test",
        gpu_count=8,
        batch_size=128,
        precision="fp32",
    )
    svc.update_job(job.id, avg_gpu_utilization=65.0, peak_gpu_memory_gib=40.0)

    profile = svc.profile_job(job.id)
    assert profile is not None
    assert profile.gpu_utilization_mean == 65.0
    assert profile.compute_efficiency > 0
    assert profile.memory_efficiency > 0
    assert len(profile.recommendations) > 0


def test_hpo():
    svc = TrainingService()
    cid = uuid4()
    job = svc.register_job(
        cluster_id=cid,
        job_name="hpo-test",
        gpu_count=8,
        batch_size=32,
        precision="fp32",
    )
    svc.update_job(job.id, throughput_samples_per_sec=100.0)

    config = HPOConfig(
        batch_sizes=[16, 32, 64],
        learning_rates=[1e-4, 3e-4],
        weight_decays=[0.0, 1e-4],
        warmup_steps=[0],
        max_trials=6,
    )
    result = svc.run_hpo(job.id, config)
    assert result is not None
    assert result.best_trial is not None
    assert len(result.all_trials) <= 6
    assert result.suggested_batch_size > 0
    assert result.suggested_learning_rate > 0


def test_distributed_config():
    config = TrainingService.suggest_distributed_config(
        total_gpus=16,
        gpu_model="NVIDIA H100-SXM-80GB",
        model_size_gb=70.0,
        per_gpu_memory_gb=80.0,
    )
    assert config.total_gpus == 16
    assert config.recommended_node_count == 2
    assert config.data_parallel_degree >= 1
    assert config.estimated_speedup_over_single > 0
    assert len(config.recommendations) > 0


def test_distributed_config_single_gpu():
    config = TrainingService.suggest_distributed_config(
        total_gpus=1,
        gpu_model="RTX 4090",
        model_size_gb=7.0,
        per_gpu_memory_gb=24.0,
    )
    assert config.total_gpus == 1
    assert config.data_parallel_degree == 1
    assert config.tensor_parallel_degree == 1
    assert config.pipeline_parallel_degree == 1


def test_generate_recommendations():
    svc = TrainingService()
    cid = uuid4()
    job = svc.register_job(
        cluster_id=cid,
        job_name="rec-test",
        gpu_count=8,
        batch_size=32,
    )
    svc.update_job(job.id, status=TrainingJobStatus.RUNNING, avg_gpu_utilization=15.0)

    recs = svc.generate_recommendations(cid)
    assert len(recs) >= 1
    assert any("low" in r.title.lower() for r in recs)


def test_slurm_connector_mock():
    from uuid import uuid4
    from gpuopt.schemas import ClusterRecord, ConnectorType
    from gpuopt.connectors.factory import build_connector

    cluster = ClusterRecord(
        id=uuid4(),
        name="slurm-mock",
        environment="sandbox",
        connector_type=ConnectorType.SLURM,
        options={"mock_slurm_data": "sandbox/mock-clusters/mock-slurm.json"},
    )
    connector = build_connector(cluster)
    assert connector is not None

    checks = connector.run_checks()
    assert len(checks) == 5
    for c in checks:
        assert c.status.value == "pass"

    telemetry = connector.collect_telemetry()
    assert telemetry.node_count == 6
    assert telemetry.gpu_count == 32

    slurm_telemetry = connector.collect_slurm_telemetry()
    assert slurm_telemetry.node_count == 6
    assert slurm_telemetry.gpu_count == 32
    assert len(slurm_telemetry.partitions) == 4
    assert len(slurm_telemetry.running_jobs) == 2
    assert len(slurm_telemetry.pending_jobs) == 2


def test_slurm_cluster_integration():
    import os
    import uuid
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_slurm_integration.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository
    get_settings.cache_clear()
    get_repository.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/api/v1/clusters", json={
            "name": f"slurm-test-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": "sandbox/mock-clusters/mock-slurm.json"},
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]

        r = client.post(f"/api/v1/clusters/{cid}/state")
        assert r.status_code == 200, r.text

        r = client.get(f"/api/v1/slurm/telemetry/{cid}")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["node_count"] == 6
        assert data["gpu_count"] == 32

        r = client.post(f"/api/v1/clusters/{cid}/recommendations")
        assert r.status_code == 200, r.text
        recs = r.json()["recommendations"]
        assert isinstance(recs, list)


def test_topology_aware_distributed_config():
    from gpuopt.schemas import GpuTopology, NodeTopology

    g1 = GpuTopology(
        node_name="gpu01",
        gpu_count=8,
        gpu_model="NVIDIA H100-SXM-80GB",
        nvlink_per_gpu=18,
        nvswitch_present=True,
    )
    g2 = GpuTopology(
        node_name="gpu02",
        gpu_count=8,
        gpu_model="NVIDIA H100-SXM-80GB",
        nvlink_per_gpu=18,
        nvswitch_present=True,
    )
    topology = NodeTopology(nodes=[g1, g2], has_nvswitch=True, recommended_tp_group_size=8)

    config = TrainingService.suggest_distributed_config(
        total_gpus=16,
        gpu_model="NVIDIA H100-SXM-80GB",
        model_size_gb=70.0,
        per_gpu_memory_gb=80.0,
        topology=topology,
    )
    assert config.total_gpus == 16
    assert config.recommended_node_count == 2
    assert config.topology_aware is True
    assert config.tp_within_node is True
    assert any("NVSwitch detected" in r for r in config.recommendations)
    assert config.estimated_speedup_over_single > 0
    assert len(config.recommendations) >= 3


def test_topology_aware_no_switch():
    from gpuopt.schemas import GpuTopology, NodeTopology

    g = GpuTopology(
        node_name="gpu01",
        gpu_count=4,
        gpu_model="NVIDIA RTX 4090",
        nvlink_per_gpu=0,
        nvswitch_present=False,
    )
    topology = NodeTopology(nodes=[g], has_nvswitch=False, recommended_tp_group_size=0)

    config = TrainingService.suggest_distributed_config(
        total_gpus=4,
        gpu_model="RTX 4090",
        model_size_gb=20.0,
        per_gpu_memory_gb=24.0,
        topology=topology,
    )
    assert config.total_gpus == 4
    assert config.topology_aware is True
    assert config.tensor_parallel_degree == 1


def test_topology_aware_cross_node():
    from gpuopt.schemas import GpuTopology, NodeTopology

    g1 = GpuTopology(node_name="gpu01", gpu_count=4, gpu_model="NVIDIA A100-80GB", nvlink_per_gpu=12, nvswitch_present=True)
    g2 = GpuTopology(node_name="gpu02", gpu_count=4, gpu_model="NVIDIA A100-80GB", nvlink_per_gpu=12, nvswitch_present=True)
    g3 = GpuTopology(node_name="gpu03", gpu_count=4, gpu_model="NVIDIA A100-80GB", nvlink_per_gpu=12, nvswitch_present=False)
    topology = NodeTopology(nodes=[g1, g2, g3], has_nvswitch=True, recommended_tp_group_size=4)

    config = TrainingService.suggest_distributed_config(
        total_gpus=12,
        gpu_model="NVIDIA A100-80GB",
        model_size_gb=160.0,
        per_gpu_memory_gb=80.0,
        topology=topology,
    )
    assert config.total_gpus >= 12
    assert config.recommended_node_count >= 2
    assert config.topology_aware is True
    assert config.tensor_parallel_degree >= 1
    assert config.pipeline_parallel_degree >= 1


def test_slurm_topology_endpoint():
    import os, uuid
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_slurm_topology.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository
    get_settings.cache_clear()
    get_repository.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/api/v1/clusters", json={
            "name": f"slurm-topo-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": "sandbox/mock-clusters/mock-slurm.json"},
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]

        client.post(f"/api/v1/clusters/{cid}/state")
        r = client.get(f"/api/v1/slurm/topology/{cid}")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "nodes" in data
        assert len(data["nodes"]) > 0


def test_slurm_monitor_endpoints():
    import os, uuid, time
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_slurm_monitor.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository
    get_settings.cache_clear()
    get_repository.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/api/v1/clusters", json={
            "name": f"slurm-mon-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "slurm",
            "options": {"mock_slurm_data": "sandbox/mock-clusters/mock-slurm.json"},
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]

        client.post(f"/api/v1/clusters/{cid}/state")

        r = client.post(f"/api/v1/slurm/monitor/start/{cid}/101")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "started"

        r = client.get(f"/api/v1/slurm/monitor/snapshot/{cid}")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "running_jobs" in data
        assert "pending_jobs" in data

        time.sleep(0.1)

        r = client.get(f"/api/v1/slurm/monitor/history/{cid}/101")
        assert r.status_code == 200, r.text
        history = r.json()
        assert isinstance(history, list)

        r = client.post(f"/api/v1/slurm/monitor/stop/{cid}/101")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "stopped"


class TestTrainingInferenceSeparation:
    def test_modules_do_not_cross_import(self):
        import sys
        for mod_name in list(sys.modules.keys()):
            if "gpuopt.inference" in mod_name:
                del sys.modules[mod_name]
        import gpuopt.training as tr
        import gpuopt.inference as inf
        tr_src = getattr(tr, "__file__", "") or ""
        inf_src = getattr(inf, "__file__", "") or ""
        assert tr_src != inf_src
        assert not any("inference" in str(v) for v in dir(tr) if callable(getattr(tr, v, None)))
