from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from ..repository import ClusterRepository
from .correlator import build_quality_flags, correlate_gpu_to_pod
from .metrics_exporter import (
    GpuMetricSnapshot,
    DcgmExporter,
    MemoryMetrics,
    ThermalPowerMetrics,
    UtilizationMetrics,
    InterconnectMetrics,
    HealthMetrics,
    get_exporter,
)

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    snapshot: GpuMetricSnapshot
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TelemetryPipeline:
    """Ingest, buffer, and forward GPU telemetry.

    Acts as the central pipeline between DCGM scrapes and the rest of
    the system (analytics, digital twin, recommendations).
    """

    def __init__(
        self,
        repository: ClusterRepository | None = None,
        buffer_size: int = 300,  # keep last 300 snapshots per GPU
    ) -> None:
        from ..dependencies import get_repository
        self.repository = repository or get_repository()
        self.exporter: DcgmExporter = get_exporter()
        self._buffer: dict[str, deque[TelemetryEvent]] = {}
        self._buffer_size = buffer_size

    # ── registration ──

    def register_cluster_gpus(self, cluster_id: UUID) -> int:
        """Register all GPUs from a cluster with the DCGM exporter."""
        state = self.repository.latest_state(cluster_id)
        if state is None:
            return 0
        count = 0
        for node in state.nodes:
            for gpu in node.gpu_devices:
                uuid = gpu.uuid or f"gpu-{node.name}-{gpu.index}"
                if gpu.model:
                    self.exporter.register_gpu(uuid, gpu.index, gpu.model, node.name)
                    count += 1
        return count

    # ── scrape / ingest ──

    def scrape_and_ingest(self) -> list[GpuMetricSnapshot]:
        """Scrape DCGM exporter and push results into the buffer."""
        snapshots = self.exporter.scrape()
        for snap in snapshots:
            key = snap.gpu_uuid
            if key not in self._buffer:
                self._buffer[key] = deque(maxlen=self._buffer_size)
            self._buffer[key].append(TelemetryEvent(snapshot=snap))
        return snapshots

    def latest_snapshots(self) -> list[GpuMetricSnapshot]:
        """Return the most recent snapshot per GPU."""
        result: list[GpuMetricSnapshot] = []
        for key, buf in self._buffer.items():
            if buf:
                result.append(buf[-1].snapshot)
        return result

    def snapshot_history(self, gpu_uuid: str, n: int = 60) -> list[TelemetryEvent]:
        """Return the last *n* events for a given GPU."""
        buf = self._buffer.get(gpu_uuid, deque())
        return list(buf)[-n:]

    def summary(self, cluster_id: UUID | None = None) -> dict[str, Any]:
        """Produce a summary of the current telemetry state."""
        snapshots = self.latest_snapshots()
        if cluster_id:
            # filter by cluster nodes
            mapping = correlate_gpu_to_pod(self.repository, cluster_id)
            node_names = {v["node_name"] for v in mapping.values()}
            snapshots = [s for s in snapshots if s.node_name in node_names]

        total = len(snapshots)
        if total == 0:
            return {"total_gpus": 0, "missing_metrics": 0, "avg_util": 0}

        missing = sum(1 for s in snapshots if s.has_missing_metrics)
        avg_util = (
            sum(s.utilization.sm_active_pct for s in snapshots) / total
        )
        avg_temp = (
            sum(s.thermal_power.gpu_temperature_celsius for s in snapshots) / total
        )
        avg_power = (
            sum(s.thermal_power.power_draw_watts for s in snapshots) / total
        )

        return {
            "total_gpus": total,
            "missing_metrics_count": missing,
            "missing_metrics_pct": round(missing / max(total, 1) * 100, 1),
            "avg_utilization_pct": round(avg_util, 1),
            "avg_temperature_celsius": round(avg_temp, 1),
            "avg_power_watts": round(avg_power, 1),
            "total_energy_joules": sum(s.thermal_power.total_energy_consumed_joules for s in snapshots),
        }


_pipeline: TelemetryPipeline | None = None


def get_pipeline() -> TelemetryPipeline:
    global _pipeline
    if _pipeline is None:
        from ..dependencies import get_repository
        _pipeline = TelemetryPipeline(repository=get_repository())
    return _pipeline