from __future__ import annotations

import random

from gpuopt.gpu_usage_inventory_schemas import (
    ClusterInventory,
    DcgmTelemetryResponse,
    GpuInventorySnapshot,
    GpuTelemetrySample,
    NodeGpuAllocation,
)


class GpuUsageInventoryService:
    def get_inventory(self, cluster_id: str | None = None) -> ClusterInventory:
        cid = cluster_id or "mock-cluster-1"
        nodes = [
            NodeGpuAllocation(
                node_name=f"{cid}-node-1",
                gpu_ids=[0, 1, 2, 3],
                allocated_to="inference",
                pods=["llama-8b-inference-7f9d6c8b9c-abc12", "llama-8b-inference-7f9d6c8b9c-def34"],
                gpu_model="NVIDIA H100-SXM-80GB",
                memory_gib=80.0,
            ),
            NodeGpuAllocation(
                node_name=f"{cid}-node-2",
                gpu_ids=[4, 5],
                allocated_to="training",
                pods=["training-job-fa45b2"],
                gpu_model="NVIDIA H100-SXM-80GB",
                memory_gib=80.0,
            ),
            NodeGpuAllocation(
                node_name=f"{cid}-node-2",
                gpu_ids=[6, 7],
                allocated_to="free",
                pods=[],
                gpu_model="NVIDIA H100-SXM-80GB",
                memory_gib=80.0,
            ),
        ]

        snapshot = GpuInventorySnapshot(
            total_gpu_capacity=8,
            total_allocatable_gpus=8,
            allocated_to_all_workloads=6,
            allocated_to_inference=2,
            estimated_free=2,
            allocation_utilization_pct=75.0,
            observed_utilization_source="mock_snapshot",
        )

        return ClusterInventory(cluster_id=cid, snapshot=snapshot, nodes=nodes)

    def get_dcgm_telemetry(self, cluster_id: str | None = None, num_gpus: int = 8) -> DcgmTelemetryResponse:
        samples = []
        for i in range(num_gpus):
            samples.append(GpuTelemetrySample(
                gpu_index=i,
                engine_util_pct=round(random.uniform(15, 85), 1),
                tensor_activity_pct=round(random.uniform(10, 75), 1),
                dram_activity_pct=round(random.uniform(20, 90), 1),
                framebuffer_used_gib=round(random.uniform(20, 60), 1),
                framebuffer_total_gib=80.0,
                power_draw_watts=round(random.uniform(100, 400), 1),
                gpu_temp_celsius=round(random.uniform(55, 85), 1),
                memory_temp_celsius=round(random.uniform(50, 80), 1),
                pcie_tx_bytes_per_sec=round(random.uniform(1e6, 2e9), 0),
                pcie_rx_bytes_per_sec=round(random.uniform(1e6, 2e9), 0),
                nvlink_tx_bytes_per_sec=round(random.uniform(1e8, 1e10), 0),
                nvlink_rx_bytes_per_sec=round(random.uniform(1e8, 1e10), 0),
                xid_errors=random.choices([0, 0, 0, 1], weights=[0.9, 0.9, 0.9, 0.1])[0],
            ))

        avg_engine = sum(s.engine_util_pct for s in samples) / max(len(samples), 1)
        summary = (
            f"DCGM telemetry for {num_gpus} GPU(s): avg engine util {avg_engine:.1f}%. "
            f"Allocation ({8} GPUs reserved) vs utilization ({avg_engine:.1f}% avg engine) "
            f"confirms that allocation != utilization."
        )

        return DcgmTelemetryResponse(
            cluster_id=cluster_id or "mock-cluster-1",
            daemonset_running=True,
            samples=samples,
            summary=summary,
        )

    def health(self) -> dict:
        return {"status": "healthy", "clusters_available": 1}
