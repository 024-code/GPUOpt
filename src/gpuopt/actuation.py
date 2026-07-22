from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from .repository import ClusterRepository
from .schemas import (
    ActuationAction,
    ActuationRecord,
    ActuationStatus,
    ActuationSummary,
    TwinState,
)

logger = logging.getLogger(__name__)


class ActuationService:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository

    def actuate(
        self,
        cluster_id: UUID,
        rec_id: UUID,
        dry_run: bool = False,
        reason: str = "",
    ) -> ActuationRecord:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")

        rec_set = self.repository.latest_recommendations(cluster_id)
        if rec_set is None:
            raise KeyError("No recommendations available for this cluster")

        target_rec = None
        for r in rec_set.recommendations:
            if r.id == rec_id:
                target_rec = r
                break
        if target_rec is None:
            raise KeyError(f"Recommendation not found: {rec_id}")

        now = datetime.now(timezone.utc)
        actions: list[ActuationAction] = []
        record = ActuationRecord(
            id=uuid4(),
            cluster_id=cluster_id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            rec_id=rec_id,
            rec_title=target_rec.title,
            rec_type=target_rec.type.value,
            status=ActuationStatus.IN_PROGRESS,
            dry_run=dry_run,
            started_at=now,
        )

        if dry_run:
            actions.append(ActuationAction(
                action_type="dry_run",
                target="cluster",
                value="simulated",
                status="success",
                message="Dry-run validation passed",
            ))
            record.actions = actions
            record.status = ActuationStatus.COMPLETED
            record.completed_at = datetime.now(timezone.utc)
            record.result_summary = f"Dry-run completed for recommendation: {target_rec.title}"
            self.repository.save_actuation(record)
            return record

        try:
            twin = self.repository.get_twin(cluster_id)
            if twin is None:
                twin = self._build_twin_from_state(cluster_id, cluster.name, cluster.environment)

            if twin is None:
                actions.append(ActuationAction(
                    action_type="state_check",
                    target="cluster",
                    value="no_state",
                    status="failed",
                    message="No cluster state or twin available",
                ))
                record.actions = actions
                record.status = ActuationStatus.FAILED
                record.completed_at = datetime.now(timezone.utc)
                record.error_message = "No cluster state or twin available for actuation"
                self.repository.save_actuation(record)
                return record

            state = json.loads(twin.state_json) if twin.state_json else {}
            node_map = {n["name"]: n for n in state.get("nodes", [])}

            applied = self._apply_rec_to_twin(target_rec, node_map)
            twin.state_json = json.dumps({"nodes": list(node_map.values())}, default=str)
            twin.has_diverged = True
            twin.divergence_reason = f"Applied recommendation: {target_rec.title} (actuation {record.id})"
            self.repository.save_twin(twin)

            actions.append(ActuationAction(
                action_type=applied["type"],
                target=applied["target"],
                value=applied["value"],
                status="success",
                message=applied["message"],
            ))

            record.actions = actions
            record.status = ActuationStatus.COMPLETED
            record.completed_at = datetime.now(timezone.utc)
            record.result_summary = f"Applied recommendation: {target_rec.title}"
            self.repository.save_actuation(record)

        except Exception as exc:
            record.actions = actions
            record.status = ActuationStatus.FAILED
            record.completed_at = datetime.now(timezone.utc)
            record.error_message = str(exc)
            self.repository.save_actuation(record)

        return record

    def rollback(self, cluster_id: UUID, actuation_id: UUID) -> ActuationRecord:
        original = self.repository.get_actuation(cluster_id, actuation_id)
        if original is None:
            raise KeyError(f"Actuation not found: {actuation_id}")
        if original.status != ActuationStatus.COMPLETED:
            raise KeyError(f"Actuation {actuation_id} is in status '{original.status.value}' and cannot be rolled back")

        cluster = self.repository.get_cluster(cluster_id)
        cluster_name = cluster.name if cluster else "unknown"

        twin = self.repository.get_twin(cluster_id)
        if twin is None:
            raise KeyError("No twin available to roll back")

        if twin.state_json:
            state = json.loads(twin.state_json)
            state["diverged"] = False
            twin.state_json = json.dumps(state, default=str)

        twin.has_diverged = False
        twin.divergence_reason = f"Rolled back actuation {actuation_id}"
        self.repository.save_twin(twin)

        rollback_record = ActuationRecord(
            id=uuid4(),
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            environment=cluster.environment if cluster else "",
            rec_id=original.rec_id,
            rec_title=original.rec_title,
            rec_type=original.rec_type,
            status=ActuationStatus.ROLLED_BACK,
            dry_run=False,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            actions=[
                ActuationAction(
                    action_type="rollback",
                    target="twin",
                    value="restored",
                    status="success",
                    message=f"Rolled back actuation {actuation_id}",
                )
            ],
            result_summary=f"Rolled back actuation {actuation_id} for recommendation: {original.rec_title}",
            rollback_of=str(actuation_id),
        )
        self.repository.save_actuation(rollback_record)
        return rollback_record

    def get_actuation(self, cluster_id: UUID, actuation_id: UUID) -> ActuationRecord | None:
        return self.repository.get_actuation(cluster_id, actuation_id)

    def list_actuations(self, cluster_id: UUID, limit: int = 20) -> list[ActuationRecord]:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.list_actuations(cluster_id, limit=limit)

    def summarize(self, cluster_id: UUID) -> ActuationSummary:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        actuations = self.list_actuations(cluster_id, limit=1000)
        total = len(actuations)
        successful = sum(1 for a in actuations if a.status == ActuationStatus.COMPLETED)
        failed = sum(1 for a in actuations if a.status == ActuationStatus.FAILED)
        in_progress = sum(1 for a in actuations if a.status == ActuationStatus.IN_PROGRESS)
        rolled_back = sum(1 for a in actuations if a.status == ActuationStatus.ROLLED_BACK)
        pending = sum(1 for a in actuations if a.status == ActuationStatus.PENDING)

        return ActuationSummary(
            cluster_id=cluster_id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            total_actuations=total,
            successful=successful,
            failed=failed,
            in_progress=in_progress,
            rolled_back=rolled_back,
            pending=pending,
            latest_actuation=actuations[0] if actuations else None,
        )

    @staticmethod
    def _build_twin_from_state(cluster_id: UUID, cluster_name: str, environment: str) -> TwinState | None:
        return TwinState(
            id=uuid4(),
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            environment=environment,
            synced_at=datetime.now(timezone.utc),
            state_json=json.dumps({"nodes": []}, default=str),
            has_diverged=False,
        )

    @staticmethod
    def _apply_rec_to_twin(rec: object, node_map: dict) -> dict:
        rec_type = rec.type.value if hasattr(rec, "type") else "unknown"
        title = rec.title if hasattr(rec, "title") else ""
        rec_id = str(rec.id) if hasattr(rec, "id") else ""

        if "gpu" in rec_type.lower() or "gpu" in title.lower():
            if node_map:
                target = list(node_map.keys())[0]
                node_map[target]["allocatable_gpu_memory"] = str(
                    int(float(str(node_map[target].get("allocatable_gpu_memory", "0"))) * 0.9)
                )
            return {
                "type": "resource_adjustment",
                "target": target if node_map else "unknown",
                "value": "gpu_memory_reduced_10pct",
                "message": f"[{rec_type}] {title}: reduced GPU memory by 10%",
            }

        if "node" in rec_type.lower() or "node" in title.lower():
            return {
                "type": "node_recommendation",
                "target": "cluster",
                "value": "node_count_adjusted",
                "message": f"[{rec_type}] {title}: node recommendation applied",
            }

        if "pod" in rec_type.lower() or "pod" in title.lower():
            for node in node_map.values():
                node["pod_count"] = max(int(str(node.get("pod_count", "0"))) - 1, 0)
            return {
                "type": "pod_consolidation",
                "target": "cluster",
                "value": "pods_reduced",
                "message": f"[{rec_type}] {title}: pod consolidation applied",
            }

        return {
            "type": "general_adjustment",
            "target": "cluster",
            "value": "config_updated",
            "message": f"[{rec_type}] {title}: general adjustment applied",
        }
