"""Seed mock telemetry data into the database for testing intelligence modules.

Usage:
    cd GPUOpt && python scripts/seed_telemetry.py

This creates 3 clusters (sandbox, staging, production) with realistic
node/GPU configurations and telemetry data so intelligence endpoints
have data to work with.
"""

import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import datetime, timezone
from uuid import UUID, uuid4

from gpuopt.config import get_settings
from gpuopt.repository import ClusterRepository
from gpuopt.schemas import (
    ClusterCreate,
    ClusterTelemetry,
    ClusterStateData,
    ConnectorType,
    GPUDeviceState,
    GPUDeviceTelemetry,
    NodeState,
    NodeTelemetry,
)


def _make_gpu_state(index: int, model: str, mem_total_gb: int, mem_used_gb: int, status: str = "healthy") -> GPUDeviceState:
    return GPUDeviceState(
        index=index,
        uuid=f"GPU-{uuid4().hex[:12]}",
        model=model,
        memory_total_bytes=mem_total_gb * 1024**3,
        memory_used_bytes=mem_used_gb * 1024**3,
        status=status,
    )


def _make_gpu_telemetry(index: int, util_pct: float, mem_used_gb: int, mem_total_gb: int,
                         temp_c: float, power_w: float, power_limit_w: float) -> GPUDeviceTelemetry:
    return GPUDeviceTelemetry(
        index=index,
        uuid=f"GPU-{uuid4().hex[:12]}",
        model="NVIDIA A100 80GB",
        memory_total_bytes=mem_total_gb * 1024**3,
        memory_used_bytes=mem_used_gb * 1024**3,
        utilization_gpu_percent=util_pct,
        utilization_memory_percent=(mem_used_gb / mem_total_gb) * 100,
        temperature_gpu_celsius=temp_c,
        power_draw_watts=power_w,
        power_limit_watts=power_limit_w,
        ecc_errors_volatile=0,
        ecc_errors_aggregate=0,
        clock_sm_mhz=1410,
        clock_mem_mhz=1593,
    )


# ── Cluster Configurations ─────────────────────────────────────

CLUSTER_CONFIGS = [
    {
        "name": "local-mock",
        "environment": "sandbox",
        "connector_type": ConnectorType.MOCK,
        "description": "Local mock GPU cluster for development.",
        "region": "us-east-1",
        "nodes": [
            {
                "name": "mock-node-1",
                "status": "Ready",
                "gpus": [
                    _make_gpu_state(0, "NVIDIA A100 80GB", 80, 12),
                    _make_gpu_state(1, "NVIDIA A100 80GB", 80, 8),
                    _make_gpu_state(2, "NVIDIA A100 80GB", 80, 4),
                    _make_gpu_state(3, "NVIDIA A100 80GB", 80, 2),
                ],
                "telemetry_gpus": [
                    _make_gpu_telemetry(0, 15.2, 12, 80, 42.5, 185, 400),
                    _make_gpu_telemetry(1, 8.7, 8, 80, 38.1, 95, 400),
                    _make_gpu_telemetry(2, 3.1, 4, 80, 35.0, 55, 400),
                    _make_gpu_telemetry(3, 1.2, 2, 80, 33.2, 30, 400),
                ],
            },
            {
                "name": "mock-node-2",
                "status": "Ready",
                "gpus": [
                    _make_gpu_state(0, "NVIDIA A100 80GB", 80, 40),
                    _make_gpu_state(1, "NVIDIA A100 80GB", 80, 35),
                    _make_gpu_state(2, "NVIDIA A100 80GB", 80, 20),
                    _make_gpu_state(3, "NVIDIA A100 80GB", 80, 5),
                ],
                "telemetry_gpus": [
                    _make_gpu_telemetry(0, 55.3, 40, 80, 62.7, 320, 400),
                    _make_gpu_telemetry(1, 42.1, 35, 80, 58.3, 280, 400),
                    _make_gpu_telemetry(2, 25.8, 20, 80, 48.9, 180, 400),
                    _make_gpu_telemetry(3, 6.4, 5, 80, 36.5, 70, 400),
                ],
            },
        ],
    },
    {
        "name": "training-cluster",
        "environment": "staging",
        "connector_type": ConnectorType.MOCK,
        "description": "Staging cluster for ML training workloads.",
        "region": "us-west-2",
        "nodes": [
            {
                "name": "train-node-1",
                "status": "Ready",
                "gpus": [
                    _make_gpu_state(0, "NVIDIA H100 80GB", 80, 72),
                    _make_gpu_state(1, "NVIDIA H100 80GB", 80, 65),
                    _make_gpu_state(2, "NVIDIA H100 80GB", 80, 70),
                    _make_gpu_state(3, "NVIDIA H100 80GB", 80, 68),
                ],
                "telemetry_gpus": [
                    _make_gpu_telemetry(0, 92.5, 72, 80, 78.2, 520, 700),
                    _make_gpu_telemetry(1, 88.3, 65, 80, 75.1, 490, 700),
                    _make_gpu_telemetry(2, 95.1, 70, 80, 80.5, 550, 700),
                    _make_gpu_telemetry(3, 90.7, 68, 80, 76.8, 510, 700),
                ],
            },
            {
                "name": "train-node-2",
                "status": "Ready",
                "gpus": [
                    _make_gpu_state(0, "NVIDIA H100 80GB", 80, 60),
                    _make_gpu_state(1, "NVIDIA H100 80GB", 80, 55),
                    _make_gpu_state(2, "NVIDIA H100 80GB", 80, 58),
                    _make_gpu_state(3, "NVIDIA H100 80GB", 80, 62),
                ],
                "telemetry_gpus": [
                    _make_gpu_telemetry(0, 78.4, 60, 80, 68.5, 420, 700),
                    _make_gpu_telemetry(1, 72.9, 55, 80, 65.2, 390, 700),
                    _make_gpu_telemetry(2, 76.2, 58, 80, 67.0, 405, 700),
                    _make_gpu_telemetry(3, 81.5, 62, 80, 69.8, 430, 700),
                ],
            },
        ],
    },
    {
        "name": "production-inference",
        "environment": "production",
        "connector_type": ConnectorType.MOCK,
        "description": "Production inference serving cluster.",
        "region": "eu-west-1",
        "nodes": [
            {
                "name": "inf-node-1",
                "status": "Ready",
                "gpus": [
                    _make_gpu_state(0, "NVIDIA A100 80GB", 80, 75),
                    _make_gpu_state(1, "NVIDIA A100 80GB", 80, 72),
                    _make_gpu_state(2, "NVIDIA A100 80GB", 80, 70),
                    _make_gpu_state(3, "NVIDIA A100 80GB", 80, 6),
                    _make_gpu_state(4, "NVIDIA A100 80GB", 80, 4),
                ],
                "telemetry_gpus": [
                    _make_gpu_telemetry(0, 97.2, 75, 80, 82.3, 380, 400),
                    _make_gpu_telemetry(1, 95.8, 72, 80, 80.1, 370, 400),
                    _make_gpu_telemetry(2, 93.4, 70, 80, 78.5, 355, 400),
                    _make_gpu_telemetry(3, 7.5, 6, 80, 37.2, 60, 400),
                    _make_gpu_telemetry(4, 3.8, 4, 80, 34.8, 40, 400),
                ],
            },
            {
                "name": "inf-node-2",
                "status": "Ready",
                "gpus": [
                    _make_gpu_state(0, "NVIDIA A100 80GB", 80, 78),
                    _make_gpu_state(1, "NVIDIA A100 80GB", 80, 76),
                    _make_gpu_state(2, "NVIDIA A100 80GB", 80, 74),
                    _make_gpu_state(3, "NVIDIA A100 80GB", 80, 2),
                ],
                "telemetry_gpus": [
                    _make_gpu_telemetry(0, 98.5, 78, 80, 84.7, 395, 400),
                    _make_gpu_telemetry(1, 96.3, 76, 80, 81.9, 385, 400),
                    _make_gpu_telemetry(2, 94.1, 74, 80, 79.2, 365, 400),
                    _make_gpu_telemetry(3, 2.1, 2, 80, 32.5, 25, 400),
                ],
            },
            {
                "name": "inf-node-3",
                "status": "NotReady",
                "gpus": [
                    _make_gpu_state(0, "NVIDIA A100 80GB", 80, 80, "failed"),
                    _make_gpu_state(1, "NVIDIA A100 80GB", 80, 80, "failed"),
                ],
                "telemetry_gpus": [
                    _make_gpu_telemetry(0, 0, 80, 80, 88.5, 450, 400),
                    _make_gpu_telemetry(1, 0, 80, 80, 86.2, 440, 400),
                ],
            },
        ],
    },
]


def _build_state(cluster_record, config: dict) -> ClusterStateData:
    """Build a ClusterStateData with nodes and telemetry from config."""
    nodes: list[NodeState] = []
    telemetry_nodes: list[NodeTelemetry] = []
    total_gpu_memory = 0

    for node_cfg in config["nodes"]:
        # GPU device states
        gpu_devices: list[GPUDeviceState] = []
        for gpu in node_cfg["gpus"]:
            gpu_devices.append(gpu)
            total_gpu_memory += gpu.memory_total_bytes

        nodes.append(NodeState(
            name=node_cfg["name"],
            status=node_cfg["status"],
            capacity={
                "cpu": "32",
                "memory": "256Gi",
                "pods": "110",
                "nvidia.com/gpu": str(len(gpu_devices)),
            },
            allocatable={
                "cpu": "31",
                "memory": "250Gi",
                "pods": "110",
                "nvidia.com/gpu": str(len(gpu_devices)),
            },
            labels={
                "kubernetes.io/hostname": node_cfg["name"],
                "node.kubernetes.io/instance-type": "gpu-standard",
                "nvidia.com/gpu.present": "true",
            },
            gpu_devices=gpu_devices,
            pod_count=len(gpu_devices),
            pod_capacity=110,
        ))

        # Telemetry
        telemetry_nodes.append(NodeTelemetry(
            node_name=node_cfg["name"],
            status=node_cfg["status"],
            cpu_usage_millicores=12000,
            cpu_capacity_millicores=32000,
            memory_usage_bytes=100 * 1024**3,
            memory_capacity_bytes=256 * 1024**3,
            pod_count=len(gpu_devices),
            pod_capacity=110,
            gpu_devices=node_cfg["telemetry_gpus"],
        ))

    now = datetime.now(timezone.utc)
    cluster_id = cluster_record.id

    telemetry = ClusterTelemetry(
        cluster_id=cluster_id,
        cluster_name=cluster_record.name,
        collected_at=now,
        node_count=len(nodes),
        gpu_count=sum(len(n["gpus"]) for n in config["nodes"]),
        nodes=telemetry_nodes,
        freshness_seconds=0.0,
    )

    state = ClusterStateData(
        cluster_id=cluster_id,
        cluster_name=cluster_record.name,
        environment=cluster_record.environment,
        collected_at=now,
        generated_at=now,
        node_count=len(nodes),
        gpu_count=telemetry.gpu_count,
        total_gpu_memory_bytes=total_gpu_memory,
        nodes=nodes,
        telemetry=telemetry,
    )

    return state


def main() -> int:
    import os

    settings = get_settings()
    repo = ClusterRepository(settings.database_path)

    print(f"Database: {settings.database_path}")
    print()

    for cfg in CLUSTER_CONFIGS:
        # Upsert the cluster
        payload = ClusterCreate(
            name=cfg["name"],
            environment=cfg["environment"],
            connector_type=cfg["connector_type"],
            description=cfg["description"],
            region=cfg.get("region"),
        )
        record = repo.upsert_cluster(payload)
        print(f"[OK] Cluster: {record.environment}/{record.name} ({record.id})")

        # Build and save state with telemetry
        state = _build_state(record, cfg)
        repo.save_state(state)

        gpu_count = sum(len(n["gpus"]) for n in cfg["nodes"])
        node_count = len(cfg["nodes"])
        print(f"  + State saved: {node_count} nodes, {gpu_count} GPUs, telemetry included")
        print()

    print("[DONE] Database seeded with live mock telemetry data.")
    print("Run the server:")
    print("  cd GPUOpt && PYTHONPATH=src uvicorn gpuopt.main:app --host 0.0.0.0 --port 8080 --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
