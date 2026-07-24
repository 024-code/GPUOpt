from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from ..repository import ClusterRepository

logger = logging.getLogger(__name__)


def correlate_gpu_to_pod(
    repo: ClusterRepository,
    cluster_id: UUID,
) -> dict[str, dict[str, Any]]:
    """Build a GPU-UUID → (node, pod) mapping for a given cluster.

    Returns a dict keyed by GPU UUID with ``node_name``, ``pod_name``,
    and ``pod_namespace`` (empty string when the GPU is unclaimed).
    """
    mapping: dict[str, dict[str, Any]] = {}
    state = repo.latest_state(cluster_id)
    if state is None:
        return mapping

    for node in state.nodes:
        for gpu in node.gpu_devices:
            if gpu.uuid:
                mapping[gpu.uuid] = {
                    "node_name": node.name,
                    "gpu_index": gpu.index,
                    "gpu_model": gpu.model or "unknown",
                    "pod_name": "",
                    "pod_namespace": "",
                }

    # overlay any known pod assignments from the cluster state
    if state.telemetry is not None:
        for tn in state.telemetry.nodes:
            for tg in tn.gpu_devices:
                key = tg.uuid
                if key in mapping:
                    mapping[key].update({
                        "utilization_gpu_pct": tg.utilization_gpu_percent,
                        "memory_used_bytes": tg.memory_used_bytes,
                        "temperature": tg.temperature_gpu_celsius,
                    })

    return mapping


def build_quality_flags(
    snapshot: dict[str, Any],
) -> dict[str, bool]:
    """Set quality flags for each metric category.

    A flag is ``True`` when the metric is present and within expected
    ranges, ``False`` when missing or clearly bogus.
    """
    return {
        "utilization_complete": (
            snapshot.get("sm_active_pct", -1) >= 0
            and snapshot.get("dram_active_pct", -1) >= 0
        ),
        "memory_complete": (
            snapshot.get("framebuffer_total_bytes", 0) > 0
        ),
        "thermal_complete": (
            snapshot.get("gpu_temperature_celsius", -1) >= 0
        ),
        "power_complete": (
            snapshot.get("power_draw_watts", -1) >= 0
        ),
        "interconnect_complete": (
            snapshot.get("pcie_tx_bytes_per_sec", -1) >= 0
        ),
        "health_complete": (
            snapshot.get("ecc_errors_volatile", -1) >= 0
        ),
    }