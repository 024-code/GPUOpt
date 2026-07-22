from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from .ml.drift_detector import DriftDetector
from .repository import ClusterRepository
from .schemas import (
    ClusterStateData,
    DriftItem,
    DriftSeverity,
    RecommendationSet,
    TwinComparison,
    TwinState,
)

logger = logging.getLogger(__name__)


class DigitalTwinService:
    def __init__(self, repository: ClusterRepository, drift_detector: DriftDetector | None = None) -> None:
        self.repository = repository
        self.drift_detector = drift_detector or DriftDetector()

    def sync_twin(self, cluster_id: UUID) -> TwinState:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        state = self.repository.latest_state(cluster_id)
        if state is None:
            raise KeyError("No cluster state available to sync twin")
        twin = TwinState(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            original_collected_at=state.collected_at,
            node_count=state.node_count,
            gpu_count=state.gpu_count,
            state_json=state.model_dump_json(),
        )
        self.repository.save_twin(twin)
        return twin

    def get_twin(self, cluster_id: UUID) -> TwinState | None:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.get_twin(cluster_id)

    def compare_twin(self, cluster_id: UUID) -> TwinComparison:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        twin = self.repository.get_twin(cluster_id)
        if twin is None:
            raise KeyError("No twin has been synced for this cluster")
        actual = self.repository.latest_state(cluster_id)
        if actual is None:
            raise KeyError("No actual cluster state available for comparison")

        self.drift_detector.set_baseline(actual)
        self.drift_detector.update(actual)

        twin_state = ClusterStateData.model_validate_json(twin.state_json)

        drifts: list[DriftItem] = []
        drifts.extend(self.drift_detector.detect_node_drift(twin_state, actual))
        drifts.extend(self.drift_detector.detect_feature_anomaly(actual))

        counts = {s.value: 0 for s in DriftSeverity}
        for d in drifts:
            sev = d.severity.value if hasattr(d.severity, "value") else d.severity
            counts[sev] = counts.get(sev, 0) + 1

        severity_rank = {
            DriftSeverity.CRITICAL.value: 5,
            DriftSeverity.HIGH.value: 4,
            DriftSeverity.MEDIUM.value: 3,
            DriftSeverity.LOW.value: 2,
            DriftSeverity.NONE.value: 1,
        }
        detected_severities = [s for s, c in counts.items() if c > 0]
        if detected_severities:
            overall_key = max(detected_severities, key=lambda s: severity_rank.get(s, 0))
            overall = DriftSeverity(overall_key)
        else:
            overall = DriftSeverity.NONE

        parts = [f"{len(drifts)} drift(s) detected"]
        if counts.get(DriftSeverity.CRITICAL.value, 0):
            parts.append(f"{counts[DriftSeverity.CRITICAL.value]} critical")
        if counts.get(DriftSeverity.HIGH.value, 0):
            parts.append(f"{counts[DriftSeverity.HIGH.value]} high")
        if counts.get(DriftSeverity.MEDIUM.value, 0):
            parts.append(f"{counts[DriftSeverity.MEDIUM.value]} medium")
        parts.append(f"overall: {overall.value}")

        return TwinComparison(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            twin_synced_at=twin.synced_at,
            actual_collected_at=actual.collected_at,
            drift_count=len(drifts),
            critical_drift_count=counts.get(DriftSeverity.CRITICAL.value, 0),
            high_drift_count=counts.get(DriftSeverity.HIGH.value, 0),
            medium_drift_count=counts.get(DriftSeverity.MEDIUM.value, 0),
            overall_drift_severity=overall,
            drifts=drifts,
            summary="; ".join(parts),
        )

    def apply_recommendation(self, cluster_id: UUID, rec_id: UUID) -> TwinState:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        twin = self.repository.get_twin(cluster_id)
        if twin is None:
            raise KeyError("No twin has been synced; sync before applying recommendations")
        rec_set = self.repository.latest_recommendations(cluster_id)
        if rec_set is None:
            raise KeyError("No recommendations available")
        rec = next((r for r in rec_set.recommendations if r.id == rec_id), None)
        if rec is None:
            raise KeyError(f"Recommendation not found: {rec_id}")

        twin_state = ClusterStateData.model_validate_json(twin.state_json)
        node_map = {n.name: n for n in twin_state.nodes}

        for action in rec.actions:
            for node_name in list(node_map.keys()):
                node = node_map[node_name]
                if node_name in action and "resize" in action.lower():
                    for gpu in node.gpu_devices:
                        gpu.memory_used_bytes = int(gpu.memory_total_bytes * 0.6)
                if "consolidate" in action.lower() and node.name in action:
                    for gpu in node.gpu_devices:
                        gpu.memory_used_bytes = 0

        twin.node_count = twin_state.node_count
        twin.gpu_count = twin_state.gpu_count
        twin.state_json = twin_state.model_dump_json()
        twin.has_diverged = True
        twin.divergence_reason = f"Applied recommendation: {rec.title}"
        self.repository.save_twin(twin)
        return twin

    def reset_twin(self, cluster_id: UUID) -> TwinState:
        return self.sync_twin(cluster_id)
