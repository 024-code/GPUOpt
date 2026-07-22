from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from .repository import ClusterRepository
from .schemas import (
    BaselineInfo,
    CheckItem,
    CheckStatus,
    ClusterStateData,
    GPUDeviceDiff,
    StateComparison,
    TraceListItem,
    TraceReplayResult,
)

logger = logging.getLogger(__name__)


class TraceService:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository

    def list_traces(self, cluster_id: UUID, limit: int = 50, offset: int = 0) -> list[TraceListItem]:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        baseline = self.repository.get_baseline(cluster_id)
        baseline_trace_id = baseline.trace_id if baseline else None
        entries = self.repository.list_traces(cluster_id, limit=limit, offset=offset)
        return [
            TraceListItem(
                id=tid,
                cluster_id=s.cluster_id,
                cluster_name=s.cluster_name,
                environment=s.environment,
                collected_at=s.collected_at,
                node_count=s.node_count,
                gpu_count=s.gpu_count,
                has_baseline=tid == baseline_trace_id,
            )
            for tid, s in entries
        ]

    def get_trace(self, cluster_id: UUID, trace_id: str) -> ClusterStateData:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        state = self.repository.get_trace(cluster_id, trace_id)
        if state is None:
            raise KeyError(f"Trace not found: {trace_id}")
        return state

    def trace_count(self, cluster_id: UUID) -> int:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.trace_count(cluster_id)

    def replay_trace(self, cluster_id: UUID, trace_id: str) -> TraceReplayResult:
        state = self.get_trace(cluster_id, trace_id)
        checks: list[CheckItem] = []

        node_count = len(state.nodes)
        checks.append(
            CheckItem(
                name="node_inventory",
                status=CheckStatus.PASS if node_count > 0 else CheckStatus.FAIL,
                message=f"Trace shows {node_count} node(s).",
                details={"node_count": node_count, "nodes": [n.name for n in state.nodes]},
            )
        )

        healthy_nodes = sum(1 for n in state.nodes if n.status == "Ready")
        if healthy_nodes < node_count:
            checks.append(
                CheckItem(
                    name="node_health",
                    status=CheckStatus.WARN,
                    message=f"{node_count - healthy_nodes} node(s) not Ready.",
                    details={"healthy": healthy_nodes, "total": node_count},
                )
            )

        gpu_count = state.gpu_count
        if gpu_count > 0:
            total_mem = state.total_gpu_memory_bytes
            used_mem = sum(
                g.memory_used_bytes for n in state.nodes for g in n.gpu_devices
            )
            util_pct = round((used_mem / total_mem) * 100, 1) if total_mem > 0 else 0.0
            checks.append(
                CheckItem(
                    name="gpu_inventory",
                    status=CheckStatus.PASS,
                    message=f"Trace shows {gpu_count} GPU(s), {util_pct}% memory utilized.",
                    details={
                        "gpu_count": gpu_count,
                        "total_memory_bytes": total_mem,
                        "used_memory_bytes": used_mem,
                        "utilization_percent": util_pct,
                    },
                )
            )
        else:
            checks.append(
                CheckItem(
                    name="gpu_inventory",
                    status=CheckStatus.WARN,
                    message="No GPUs detected in trace.",
                    details={"gpu_count": 0},
                )
            )

        if gpu_count > 0:
            total_temp = sum(
                g.temperature_gpu_celsius for n in state.nodes for g in n.gpu_devices if g.temperature_gpu_celsius > 0
            )
            gpus_with_temp = sum(1 for n in state.nodes for g in n.gpu_devices if g.temperature_gpu_celsius > 0)
            avg_temp = round(total_temp / gpus_with_temp, 1) if gpus_with_temp > 0 else 0.0
            total_power = sum(
                g.power_draw_watts for n in state.nodes for g in n.gpu_devices
            )
            checks.append(
                CheckItem(
                    name="gpu_telemetry",
                    status=CheckStatus.PASS if gpus_with_temp > 0 else CheckStatus.WARN,
                    message=f"Avg GPU temp: {avg_temp}°C, total power: {total_power}W.",
                    details={
                        "avg_temperature_celsius": avg_temp,
                        "total_power_watts": total_power,
                        "gpus_with_telemetry": gpus_with_temp,
                    },
                )
            )

        total_pods = sum(n.pod_count for n in state.nodes)
        total_capacity = sum(n.pod_capacity for n in state.nodes if n.pod_capacity > 0)
        checks.append(
            CheckItem(
                name="workload_summary",
                status=CheckStatus.PASS,
                message=f"Total pods: {total_pods} across {node_count} node(s).",
                details={
                    "total_pods": total_pods,
                    "total_pod_capacity": total_capacity,
                    "node_count": node_count,
                },
            )
        )

        if state.telemetry and state.telemetry.freshness_seconds > 0:
            checks.append(
                CheckItem(
                    name="telemetry_freshness",
                    status=CheckStatus.PASS if state.telemetry.freshness_seconds < 300 else CheckStatus.WARN,
                    message=f"Telemetry freshness: {state.telemetry.freshness_seconds}s.",
                    details={"freshness_seconds": state.telemetry.freshness_seconds},
                )
            )

        overall = CheckStatus.PASS
        if any(c.status == CheckStatus.FAIL for c in checks):
            overall = CheckStatus.FAIL
        elif any(c.status == CheckStatus.WARN for c in checks):
            overall = CheckStatus.WARN

        summary = {s.value: sum(c.status == s for c in checks) for s in CheckStatus}

        return TraceReplayResult(
            trace_id=trace_id,
            cluster_id=state.cluster_id,
            cluster_name=state.cluster_name,
            environment=state.environment,
            original_collected_at=state.collected_at,
            node_count=state.node_count,
            gpu_count=state.gpu_count,
            checks=checks,
            overall_status=overall,
            summary=summary,
        )

    def set_baseline(self, cluster_id: UUID) -> BaselineInfo:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        entries = self.repository.list_traces(cluster_id, limit=1)
        if not entries:
            raise KeyError("No cluster state has been collected; cannot set baseline")
        trace_id, state = entries[0]
        return self.repository.set_baseline(cluster_id, state, trace_id)

    def get_baseline(self, cluster_id: UUID) -> BaselineInfo:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        baseline = self.repository.get_baseline(cluster_id)
        if baseline is None:
            raise KeyError("No baseline has been set for this cluster")
        return baseline

    def _get_current_trace_id(self, cluster_id: UUID) -> str | None:
        entries = self.repository.list_traces(cluster_id, limit=1)
        return entries[0][0] if entries else None

    def compare_with_baseline(self, cluster_id: UUID) -> StateComparison:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        baseline_info = self.repository.get_baseline(cluster_id)
        if baseline_info is None:
            raise KeyError("No baseline has been set for this cluster")
        baseline_state = self.repository.get_trace(cluster_id, baseline_info.trace_id)
        if baseline_state is None:
            raise KeyError("Baseline trace data no longer available")
        current_state = self.repository.latest_state(cluster_id)
        if current_state is None:
            raise KeyError("No current state available for comparison")
        current_id = self._get_current_trace_id(cluster_id) or ""
        return self._compare_states(baseline_state, current_state, baseline_info.trace_id, current_id)

    def compare_traces(self, cluster_id: UUID, trace_id_a: str, trace_id_b: str) -> StateComparison:
        state_a = self.get_trace(cluster_id, trace_id_a)
        state_b = self.get_trace(cluster_id, trace_id_b)
        return self._compare_states(state_a, state_b, trace_id_a, trace_id_b)

    @staticmethod
    def _compare_states(baseline: ClusterStateData, current: ClusterStateData,
                        baseline_id: str = "", current_id: str = "") -> StateComparison:
        baseline_nodes = {n.name: n for n in baseline.nodes}
        current_nodes = {n.name: n for n in current.nodes}

        baseline_node_names = set(baseline_nodes.keys())
        current_node_names = set(current_nodes.keys())

        nodes_added = sorted(current_node_names - baseline_node_names)
        nodes_removed = sorted(baseline_node_names - current_node_names)

        gpu_diffs: list[GPUDeviceDiff] = []
        common_nodes = baseline_node_names & current_node_names

        for node_name in sorted(common_nodes):
            bn = baseline_nodes[node_name]
            cn = current_nodes[node_name]
            baseline_gpus = {g.index: g for g in bn.gpu_devices}
            current_gpus = {g.index: g for g in cn.gpu_devices}
            all_indices = sorted(set(baseline_gpus.keys()) | set(current_gpus.keys()))
            for idx in all_indices:
                bg = baseline_gpus.get(idx)
                cg = current_gpus.get(idx)
                b_used = bg.memory_used_bytes if bg else 0
                c_used = cg.memory_used_bytes if cg else 0
                b_util = bg.utilization_gpu_percent if bg else 0.0
                c_util = cg.utilization_gpu_percent if cg else 0.0
                b_temp = bg.temperature_gpu_celsius if bg else 0.0
                c_temp = cg.temperature_gpu_celsius if cg else 0.0

                mem_drift = abs(c_used - b_used) / max(b_used, 1)
                util_drift = abs(c_util - b_util) / 100.0
                temp_drift = abs(c_temp - b_temp) / max(b_temp, 1) if b_temp > 0 else 0.0
                drift = round((mem_drift + util_drift + temp_drift) / 3.0, 4)

                gpu_diffs.append(
                    GPUDeviceDiff(
                        node=node_name,
                        gpu_index=idx,
                        gpu_uuid=(cg or bg).uuid or "",
                        gpu_model=(cg or bg).model or "",
                        baseline_memory_used_bytes=b_used,
                        current_memory_used_bytes=c_used,
                        baseline_utilization_percent=b_util,
                        current_utilization_percent=c_util,
                        baseline_temperature_celsius=b_temp,
                        current_temperature_celsius=c_temp,
                        drift_score=drift,
                    )
                )

        avg_drift = round(sum(d.drift_score for d in gpu_diffs) / max(len(gpu_diffs), 1), 4)
        max_drift = round(max((d.drift_score for d in gpu_diffs), default=0.0), 4)
        elapsed = (current.collected_at - baseline.collected_at).total_seconds() / 3600.0

        drift_desc = "stable"
        if max_drift > 0.5:
            drift_desc = "significant drift detected"
        elif max_drift > 0.2:
            drift_desc = "moderate drift detected"

        parts = []
        if nodes_added:
            parts.append(f"{len(nodes_added)} node(s) added")
        if nodes_removed:
            parts.append(f"{len(nodes_removed)} node(s) removed")
        if gpu_diffs:
            parts.append(f"avg GPU drift: {avg_drift:.2%}")
        parts.append(drift_desc)

        return StateComparison(
            baseline_id=baseline_id,
            current_id=current_id,
            cluster_id=baseline.cluster_id,
            cluster_name=baseline.cluster_name,
            environment=baseline.environment,
            baseline_collected_at=baseline.collected_at,
            current_collected_at=current.collected_at,
            elapsed_hours=round(elapsed, 2),
            node_count_baseline=baseline.node_count,
            node_count_current=current.node_count,
            nodes_added=nodes_added,
            nodes_removed=nodes_removed,
            gpu_count_baseline=baseline.gpu_count,
            gpu_count_current=current.gpu_count,
            gpu_diffs=gpu_diffs,
            avg_gpu_drift_score=avg_drift,
            max_gpu_drift_score=max_drift,
            summary="; ".join(parts),
        )
