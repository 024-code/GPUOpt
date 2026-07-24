from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

DCGM_SAMPLE_RATE = 15  # seconds between metric collection cycles


# ── Metric categories ──

@dataclass
class UtilizationMetrics:
    sm_active_pct: float = 0.0
    tensor_active_pct: float = 0.0
    dram_active_pct: float = 0.0
    fp32_active_pct: float = 0.0
    fp16_active_pct: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.sm_active_pct > 0 or self.tensor_active_pct > 0


@dataclass
class MemoryMetrics:
    framebuffer_used_bytes: int = 0
    framebuffer_total_bytes: int = 0
    memory_temperature_celsius: float = 0.0
    framebuffer_used_pct: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.framebuffer_total_bytes > 0


@dataclass
class ThermalPowerMetrics:
    gpu_temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    power_limit_watts: float = 0.0
    total_energy_consumed_joules: float = 0.0
    fan_speed_pct: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.power_draw_watts > 0 or self.gpu_temperature_celsius > 0


@dataclass
class InterconnectMetrics:
    pcie_tx_bytes_per_sec: int = 0
    pcie_rx_bytes_per_sec: int = 0
    pcie_replay_counter: int = 0
    nvlink_tx_bytes_per_sec: dict[int, int] = field(default_factory=dict)
    nvlink_rx_bytes_per_sec: dict[int, int] = field(default_factory=dict)
    nvlink_replay_counter: dict[int, int] = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return self.pcie_tx_bytes_per_sec > 0 or self.pcie_rx_bytes_per_sec > 0


@dataclass
class HealthMetrics:
    xid_errors: list[dict[str, Any]] = field(default_factory=list)
    ecc_errors_volatile: int = 0
    ecc_errors_aggregate: int = 0
    retired_pages_total: int = 0
    retired_pages_pending: int = 0
    last_health_event_at: str | None = None

    @property
    def has_data(self) -> bool:
        return bool(self.xid_errors) or self.ecc_errors_volatile > 0


@dataclass
class MigInstance:
    gi_profile: str = ""
    gi_instance_id: int = 0
    ci_profile: str = ""
    ci_instance_id: int = 0
    memory_total_bytes: int = 0
    memory_used_bytes: int = 0
    sm_pct: float = 0.0

    @property
    def allocatable(self) -> bool:
        return bool(self.gi_profile)


@dataclass
class GpuMetricSnapshot:
    gpu_uuid: str
    gpu_index: int
    gpu_model: str
    node_name: str
    pod_name: str = ""
    pod_namespace: str = ""
    utilization: UtilizationMetrics = field(default_factory=UtilizationMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    thermal_power: ThermalPowerMetrics = field(default_factory=ThermalPowerMetrics)
    interconnect: InterconnectMetrics = field(default_factory=InterconnectMetrics)
    health: HealthMetrics = field(default_factory=HealthMetrics)
    mig_instances: list[MigInstance] = field(default_factory=list)
    quality_flags: dict[str, bool] = field(default_factory=lambda: defaultdict(lambda: True))
    collected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def has_missing_metrics(self) -> bool:
        """True when any required category is missing data."""
        return not (self.utilization.has_data or self.memory.has_data)


# ── DCGM Exporter simulation ──

class DcgmExporter:
    """Simulates a DCGM Exporter's /metrics endpoint.

    In production this would be replaced by a real scrape of
    ``dcgm-exporter:9400/metrics``.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._gpu_models = [
            "NVIDIA A100-SXM-80GB",
            "NVIDIA A100-PCIE-40GB",
            "NVIDIA H100-SXM-80GB",
            "NVIDIA H200-SXM-141GB",
            "NVIDIA L40S",
            "NVIDIA RTX 4090",
        ]
        self._simulated_gpus: dict[str, dict[str, Any]] = {}

    def register_gpu(self, uuid: str, index: int, model: str, node: str) -> None:
        self._simulated_gpus[uuid] = {
            "uuid": uuid, "index": index, "model": model,
            "node": node,
        }

    def scrape(self) -> list[GpuMetricSnapshot]:
        now = datetime.now(timezone.utc).isoformat()
        snapshots: list[GpuMetricSnapshot] = []

        for gpu_data in self._simulated_gpus.values():
            sm = self._rng.uniform(5, 98)
            snap = GpuMetricSnapshot(
                gpu_uuid=gpu_data["uuid"],
                gpu_index=gpu_data["index"],
                gpu_model=gpu_data["model"],
                node_name=gpu_data["node"],
                utilization=UtilizationMetrics(
                    sm_active_pct=sm,
                    tensor_active_pct=self._rng.uniform(0, sm),
                    dram_active_pct=self._rng.uniform(0, sm),
                ),
                memory=MemoryMetrics(
                    framebuffer_total_bytes=85_899_345_920,
                    framebuffer_used_bytes=int(85_899_345_920 * self._rng.uniform(0.05, 0.95)),
                    memory_temperature_celsius=self._rng.uniform(30, 85),
                    framebuffer_used_pct=0.0,
                ),
                thermal_power=ThermalPowerMetrics(
                    gpu_temperature_celsius=self._rng.uniform(30, 90),
                    power_draw_watts=self._rng.uniform(50, 400),
                    power_limit_watts=400.0,
                    total_energy_consumed_joules=self._rng.uniform(1e6, 1e8),
                ),
                interconnect=InterconnectMetrics(
                    pcie_tx_bytes_per_sec=self._rng.randint(0, 10**9),
                    pcie_rx_bytes_per_sec=self._rng.randint(0, 10**9),
                    pcie_replay_counter=self._rng.randint(0, 5),
                ),
                health=HealthMetrics(
                    xid_errors=[],
                    ecc_errors_volatile=self._rng.randint(0, 20),
                    ecc_errors_aggregate=self._rng.randint(0, 100),
                    retired_pages_total=self._rng.randint(0, 10),
                ),
                collected_at=now,
            )
            snap.memory.framebuffer_used_pct = round(
                snap.memory.framebuffer_used_bytes / max(snap.memory.framebuffer_total_bytes, 1) * 100, 1
            )
            snapshots.append(snap)

        return snapshots


# ── Global exporter singleton ──

_exporter: DcgmExporter | None = None


def get_exporter() -> DcgmExporter:
    global _exporter
    if _exporter is None:
        _exporter = DcgmExporter()
    return _exporter


def format_prometheus_metrics(snapshots: list[GpuMetricSnapshot]) -> str:
    """Render GPU metric snapshots as Prometheus text exposition format."""
    lines: list[str] = [
        "# HELP gpuopt_dcgm_simulation Simulated DCGM Exporter metrics",
        "# TYPE gpuopt_dcgm_simulation gauge",
    ]
    ts = int(time.time())
    for s in snapshots:
        labels = f'gpu_uuid="{s.gpu_uuid}",gpu_index="{s.gpu_index}",model="{s.gpu_model}",node="{s.node_name}"'
        lines.append(f'gpu_utilization_sm_active_pct{{{labels}}} {s.utilization.sm_active_pct} {ts}')
        lines.append(f'gpu_utilization_tensor_active_pct{{{labels}}} {s.utilization.tensor_active_pct} {ts}')
        lines.append(f'gpu_utilization_dram_active_pct{{{labels}}} {s.utilization.dram_active_pct} {ts}')
        lines.append(f'gpu_memory_framebuffer_used_bytes{{{labels}}} {s.memory.framebuffer_used_bytes} {ts}')
        lines.append(f'gpu_memory_framebuffer_total_bytes{{{labels}}} {s.memory.framebuffer_total_bytes} {ts}')
        lines.append(f'gpu_memory_temperature_celsius{{{labels}}} {s.memory.memory_temperature_celsius} {ts}')
        lines.append(f'gpu_temperature_celsius{{{labels}}} {s.thermal_power.gpu_temperature_celsius} {ts}')
        lines.append(f'gpu_power_draw_watts{{{labels}}} {s.thermal_power.power_draw_watts} {ts}')
        lines.append(f'gpu_power_limit_watts{{{labels}}} {s.thermal_power.power_limit_watts} {ts}')
        lines.append(f'gpu_energy_consumed_joules{{{labels}}} {s.thermal_power.total_energy_consumed_joules} {ts}')
        lines.append(f'gpu_pcie_tx_bytes{{{labels}}} {s.interconnect.pcie_tx_bytes_per_sec} {ts}')
        lines.append(f'gpu_pcie_rx_bytes{{{labels}}} {s.interconnect.pcie_rx_bytes_per_sec} {ts}')
        lines.append(f'gpu_pcie_replay_counter{{{labels}}} {s.interconnect.pcie_replay_counter} {ts}')
        lines.append(f'gpu_ecc_errors_volatile{{{labels}}} {s.health.ecc_errors_volatile} {ts}')
        lines.append(f'gpu_ecc_errors_aggregate{{{labels}}} {s.health.ecc_errors_aggregate} {ts}')
        lines.append(f'gpu_retired_pages_total{{{labels}}} {s.health.retired_pages_total} {ts}')
        lines.append(f'gpu_quality_has_data{{{labels}}} {1 if not s.has_missing_metrics else 0} {ts}')
        for mi in s.mig_instances:
            mig_labels = f'{labels},gi_profile="{mi.gi_profile}",gi_instance="{mi.gi_instance_id}"'
            lines.append(f'gpu_mig_memory_used_bytes{{{mig_labels}}} {mi.memory_used_bytes} {ts}')
            lines.append(f'gpu_mig_sm_pct{{{mig_labels}}} {mi.sm_pct} {ts}')
    return "\n".join(lines)