from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any

from gpuopt.dcgm_ingestion import DcgmIngestionPipeline, get_dcgm_pipeline
from gpuopt.dcgm_quality import DcgmQualityAnalyzer
from gpuopt.gpu_usage_inventory_schemas import (
    ClusterInventory,
    DcgmTelemetryResponse,
    GpuInventorySnapshot,
    GpuTelemetrySample,
    NodeGpuAllocation,
)

logger = logging.getLogger(__name__)


class GpuUsageInventoryService:
    def __init__(self, dcgm_pipeline: DcgmIngestionPipeline | None = None) -> None:
        self._dcgm_pipeline = dcgm_pipeline or get_dcgm_pipeline()
        self._quality_analyzer = DcgmQualityAnalyzer(self._dcgm_pipeline)

    def get_inventory(self, cluster_id: str | None = None) -> ClusterInventory:
        cid = cluster_id or "mock-cluster-1"
        dcgm_samples = self._dcgm_pipeline.build_telemetry_samples()
        if dcgm_samples:
            nodes = self._build_from_dcgm(cid, dcgm_samples)
        else:
            nodes = self._mock_nodes(cid)

        gpu_count = sum(len(n.gpu_ids) for n in nodes)
        allocated = sum(1 for n in nodes if n.allocated_to != "free")
        snapshot = GpuInventorySnapshot(
            total_gpu_capacity=gpu_count,
            total_allocatable_gpus=gpu_count,
            allocated_to_all_workloads=allocated,
            allocated_to_inference=sum(1 for n in nodes if n.allocated_to == "inference"),
            estimated_free=gpu_count - allocated,
            allocation_utilization_pct=round(allocated / max(gpu_count, 1) * 100, 1),
            observed_utilization_source="dcgm_exporter" if dcgm_samples else "mock_snapshot",
        )

        return ClusterInventory(cluster_id=cid, snapshot=snapshot, nodes=nodes)

    def _build_from_dcgm(self, cluster_id: str,
                         dcgm_samples: list[dict[str, Any]]) -> list[NodeGpuAllocation]:
        gpu_count = len(dcgm_samples)
        node = NodeGpuAllocation(
            node_name=f"{cluster_id}-node-1",
            gpu_ids=list(range(gpu_count)),
            allocated_to="inference" if gpu_count > 0 else "free",
            pods=[f"gpuopt-workload-{i}" for i in range(min(gpu_count, 4))],
            gpu_model="NVIDIA GPU (DCGM)",
            memory_gib=float(dcgm_samples[0].get("fb_total", 80)) / 1024 if dcgm_samples else 80.0,
        )
        return [node]

    def _mock_nodes(self, cluster_id: str) -> list[NodeGpuAllocation]:
        return [
            NodeGpuAllocation(
                node_name=f"{cluster_id}-node-1", gpu_ids=[0, 1, 2, 3],
                allocated_to="inference",
                pods=["llama-8b-inference-7f9d6c8b9c-abc12", "llama-8b-inference-7f9d6c8b9c-def34"],
                gpu_model="NVIDIA H100-SXM-80GB", memory_gib=80.0,
            ),
            NodeGpuAllocation(
                node_name=f"{cluster_id}-node-2", gpu_ids=[4, 5],
                allocated_to="training", pods=["training-job-fa45b2"],
                gpu_model="NVIDIA H100-SXM-80GB", memory_gib=80.0,
            ),
            NodeGpuAllocation(
                node_name=f"{cluster_id}-node-2", gpu_ids=[6, 7],
                allocated_to="free", pods=[],
                gpu_model="NVIDIA H100-SXM-80GB", memory_gib=80.0,
            ),
        ]

    def get_dcgm_telemetry(self, cluster_id: str | None = None,
                           num_gpus: int = 8) -> DcgmTelemetryResponse:
        dcgm_samples = self._dcgm_pipeline.build_telemetry_samples()
        if dcgm_samples:
            samples = []
            for s in dcgm_samples[:num_gpus]:
                samples.append(GpuTelemetrySample(
                    gpu_index=s.get("gpu_index", 0),
                    engine_util_pct=float(s.get("dcgm_fi_dev_gpu_util", 0)),
                    tensor_activity_pct=float(s.get("dcgm_fi_dev_tensor_activity", 0)),
                    dram_activity_pct=float(s.get("dcgm_fi_dev_dram_activity", 0)),
                    framebuffer_used_gib=float(s.get("dcgm_fi_dev_fb_used", 0)) / 1024,
                    framebuffer_total_gib=80.0,
                    power_draw_watts=float(s.get("dcgm_fi_dev_power_usage", 0)),
                    gpu_temp_celsius=float(s.get("dcgm_fi_dev_gpu_temp", 0)),
                    memory_temp_celsius=float(s.get("dcgm_fi_dev_mem_max_temp", 0)),
                    pcie_tx_bytes_per_sec=float(s.get("dcgm_fi_dev_pcie_tx_throughput", 0)),
                    pcie_rx_bytes_per_sec=float(s.get("dcgm_fi_dev_pcie_rx_throughput", 0)),
                    nvlink_tx_bytes_per_sec=float(s.get("dcgm_fi_dev_nvlink_tx_throughput", 0)),
                    nvlink_rx_bytes_per_sec=float(s.get("dcgm_fi_dev_nvlink_rx_throughput", 0)),
                    xid_errors=int(float(s.get("dcgm_fi_dev_xid_errors", 0))),
                ))
            avg_engine = sum(s.engine_util_pct for s in samples) / max(len(samples), 1)
            return DcgmTelemetryResponse(
                cluster_id=cluster_id or "dcgm-cluster",
                daemonset_running=True,
                samples=samples,
                summary=f"DCGM telemetry for {len(samples)} GPU(s): avg engine util {avg_engine:.1f}%. "
                        f"Data source: DCGM exporter.",
            )

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
        return DcgmTelemetryResponse(
            cluster_id=cluster_id or "mock-cluster-1",
            daemonset_running=True, samples=samples,
            summary=f"DCGM telemetry for {num_gpus} GPU(s): avg engine util {avg_engine:.1f}%.",
        )

    def get_dcgm_quality_report(self, cluster_id: str = "dcgm-cluster") -> dict:
        report = self._quality_analyzer.analyze_cluster(cluster_id, self._dcgm_pipeline)
        return {
            "cluster_id": report.cluster_id,
            "total_gpus": report.total_gpus,
            "healthy_gpus": report.healthy_gpus,
            "warning_gpus": report.warning_gpus,
            "critical_gpus": report.critical_gpus,
            "overall_quality": report.overall_quality.value,
            "quality_score": report.quality_score,
            "gpu_flags": [
                {
                    "gpu_index": f.gpu_index,
                    "gpu_uuid": f.gpu_uuid,
                    "data_quality": f.data_quality.value,
                    "quality_score": f.quality_score,
                    "warnings": f.warnings,
                    "xid_errors_recent": f.xid_errors_recent,
                    "ecc_errors_aggregate": f.ecc_errors_aggregate,
                    "temperature_warning": f.temperature_warning,
                    "power_warning": f.power_warning,
                }
                for f in report.gpu_flags
            ],
            "generated_at": report.generated_at,
        }

    def health(self) -> dict:
        return {"status": "healthy", "clusters_available": 1}
