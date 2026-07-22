from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import numpy as np

from gpuopt.schemas import (
    ClusterStateData,
    GPUDeviceTelemetry,
    NodeEfficiency,
    NodeTelemetry,
    ResourceRecommendation,
    WorkloadAnalysisResult,
)


def extract_state_features(state: ClusterStateData) -> dict[str, float]:
    nodes = state.nodes
    gpu_devices = [g for n in nodes for g in n.gpu_devices]
    total_gpus = len(gpu_devices)
    total_mem = sum(g.memory_total_bytes for g in gpu_devices)
    used_mem = sum(g.memory_used_bytes for g in gpu_devices)
    mem_utils = [
        g.memory_used_bytes / max(g.memory_total_bytes, 1) * 100
        for g in gpu_devices if g.memory_total_bytes > 0
    ]

    features: dict[str, float] = {
        "node_count": float(len(nodes)),
        "gpu_count": float(total_gpus),
        "total_memory_gb": float(total_mem / (1024**3)),
        "used_memory_gb": float(used_mem / (1024**3)),
        "memory_utilization_pct": float(np.mean(mem_utils)) if mem_utils else 0.0,
        "memory_utilization_std": float(np.std(mem_utils)) if len(mem_utils) > 1 else 0.0,
        "memory_utilization_max": float(np.max(mem_utils)) if mem_utils else 0.0,
        "memory_utilization_min": float(np.min(mem_utils)) if mem_utils else 0.0,
        "gpu_idle_count": float(sum(1 for g in gpu_devices if g.memory_used_bytes < g.memory_total_bytes * 0.1)),
        "gpu_hot_count": float(sum(1 for g in gpu_devices if g.memory_used_bytes > g.memory_total_bytes * 0.85)),
        "node_ready_ratio": float(
            sum(1 for n in nodes if n.status == "Ready") / max(len(nodes), 1)
        ),
        "pod_density": float(
            sum(n.pod_count for n in nodes) / max(sum(n.pod_capacity for n in nodes), 1)
        ),
        "memory_fragmentation": float(
            np.std([g.memory_used_bytes / max(g.memory_total_bytes, 1) for g in gpu_devices])
            if len(gpu_devices) > 1 else 0.0
        ),
    }
    return features


def extract_telemetry_features(telemetry: Any) -> dict[str, float]:
    if telemetry is None:
        return {}
    gpu_devices: list[GPUDeviceTelemetry] = []
    for node in getattr(telemetry, "nodes", []):
        gpu_devices.extend(node.gpu_devices)

    utils = [g.utilization_gpu_percent for g in gpu_devices]
    mem_utils = [
        g.memory_used_bytes / max(g.memory_total_bytes, 1) * 100
        for g in gpu_devices if g.memory_total_bytes > 0
    ]
    temps = [g.temperature_gpu_celsius for g in gpu_devices if g.temperature_gpu_celsius > 0]
    powers = [g.power_draw_watts for g in gpu_devices]

    features: dict[str, float] = {
        "avg_gpu_utilization": float(np.mean(utils)) if utils else 0.0,
        "max_gpu_utilization": float(np.max(utils)) if utils else 0.0,
        "std_gpu_utilization": float(np.std(utils)) if len(utils) > 1 else 0.0,
        "avg_memory_utilization": float(np.mean(mem_utils)) if mem_utils else 0.0,
        "max_memory_utilization": float(np.max(mem_utils)) if mem_utils else 0.0,
        "avg_temperature": float(np.mean(temps)) if temps else 0.0,
        "max_temperature": float(np.max(temps)) if temps else 0.0,
        "total_power_watts": float(np.sum(powers)),
        "avg_power_watts": float(np.mean(powers)) if powers else 0.0,
        "gpu_utilization_variance": float(np.var(utils)) if len(utils) > 1 else 0.0,
    }
    return features


def extract_analysis_features(analysis: WorkloadAnalysisResult) -> dict[str, float]:
    features: dict[str, float] = {
        "overall_efficiency_score": float(analysis.overall_efficiency_score),
        "total_idle_gpu_hours": float(analysis.total_idle_gpu_hours),
        "estimated_power_waste_kwh": float(analysis.estimated_power_waste_kwh),
        "trace_count": float(analysis.trace_count),
        "timeframe_hours": float(analysis.timeframe_hours),
        "total_gpu_hours": float(analysis.total_gpu_hours),
    }
    if analysis.gpu_trends:
        idle_pcts = [t.idle_percent for t in analysis.gpu_trends]
        mem_pressures = [t.memory_pressure_percent for t in analysis.gpu_trends]
        features.update({
            "avg_gpu_idle_pct": float(np.mean(idle_pcts)),
            "max_gpu_idle_pct": float(np.max(idle_pcts)),
            "avg_memory_pressure_pct": float(np.mean(mem_pressures)),
            "max_memory_pressure_pct": float(np.max(mem_pressures)),
        })
    if analysis.node_efficiencies:
        scores = [n.efficiency_score for n in analysis.node_efficiencies]
        features.update({
            "min_node_efficiency": float(np.min(scores)),
            "max_node_efficiency": float(np.max(scores)),
            "std_node_efficiency": float(np.std(scores)) if len(scores) > 1 else 0.0,
        })
    return features


def extract_rec_features(rec: ResourceRecommendation) -> dict[str, float]:
    severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    type_map = {
        "placement": 0, "right_sizing": 1, "scaling": 2,
        "risk_mitigation": 3, "efficiency": 4,
    }
    risk_map = {"high": 2, "medium": 1, "low": 0}
    total_savings = 0.0
    for v in rec.estimated_savings.values():
        if isinstance(v, (int, float)):
            total_savings += abs(v)
    features: dict[str, float] = {
        "severity_encoded": float(severity_map.get(rec.severity.value, 0)),
        "type_encoded": float(type_map.get(rec.type.value, 0)),
        "confidence": float(rec.confidence),
        "risk_level_encoded": float(risk_map.get(rec.risk_level, 0)),
        "total_estimated_savings": float(total_savings),
        "action_count": float(len(rec.actions)),
        "affected_resource_count": float(len(rec.affected_resources)),
        "current_score": float(rec.score),
    }
    return features


def build_time_series_matrix(
    states: list[ClusterStateData],
) -> tuple[np.ndarray, list[datetime]]:
    timestamps: list[datetime] = []
    rows: list[list[float]] = []
    for state in states:
        timestamps.append(state.collected_at)
        feats = extract_state_features(state)
        telemetry = getattr(state, "telemetry", None)
        if telemetry:
            feats.update(extract_telemetry_features(telemetry))
        rows.append([feats[k] for k in sorted(feats.keys())])
    if not rows:
        return np.array([]), []
    return np.array(rows), timestamps
