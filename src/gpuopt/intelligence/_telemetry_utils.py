from __future__ import annotations

from typing import Any

from ..schemas import ClusterStateData


def telemetry_map(state: ClusterStateData) -> dict[tuple[str, int], dict[str, Any]]:
    """Build a (node_name, gpu_index) → merged-GPU-data dictionary.

    Merges the rich GPUDeviceTelemetry fields on top of the basic
    GPUDeviceState fields.  When telemetry is missing every GPU still
    gets a default entry so callers never need to guard against lookups
    returning ``None``.
    """
    result: dict[tuple[str, int], dict[str, Any]] = {}

    # basic state data (always available)
    for node in state.nodes:
        for gpu in node.gpu_devices:
            key = (node.name, gpu.index)
            result[key] = {
                "index": gpu.index,
                "uuid": gpu.uuid,
                "model": gpu.model,
                "memory_total_bytes": gpu.memory_total_bytes,
                "memory_used_bytes": gpu.memory_used_bytes,
                "status": gpu.status,
                "gpu_util_pct": 0.0,
                "mem_util_pct": 0.0,
                "temperature_celsius": 0.0,
                "power_watts": 0.0,
                "power_limit_watts": 0.0,
                "ecc_errors_total": 0,
                "ecc_errors_aggregate": 0,
                "clock_sm_mhz": 0,
                "clock_mem_mhz": 0,
            }

    # overlay telemetry when available
    if state.telemetry is not None:
        for tn in state.telemetry.nodes:
            for tg in tn.gpu_devices:
                key = (tn.node_name, tg.index)
                if key in result:
                    result[key].update({
                        "gpu_util_pct": tg.utilization_gpu_percent,
                        "mem_util_pct": tg.utilization_memory_percent,
                        "temperature_celsius": tg.temperature_gpu_celsius,
                        "power_watts": tg.power_draw_watts,
                        "power_limit_watts": tg.power_limit_watts,
                        "ecc_errors_total": tg.ecc_errors_volatile,
                        "ecc_errors_aggregate": tg.ecc_errors_aggregate,
                        "clock_sm_mhz": tg.clock_sm_mhz,
                        "clock_mem_mhz": tg.clock_mem_mhz,
                    })

    return result
