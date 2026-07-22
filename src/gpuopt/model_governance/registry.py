from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .models import (
    ModelActionClass,
    ModelMetadata,
    ModelStatus,
    ModelVersion,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._versions: dict[UUID, ModelVersion] = {}
        self._champions: dict[str, UUID] = {}     # action_class -> version_id
        self._challengers: dict[str, UUID] = {}    # action_class -> version_id
        self._prediction_counts: dict[UUID, int] = {}
        self._confidence_sum: dict[UUID, float] = {}
        self._confidence_count: dict[UUID, int] = {}
        self._fallback_active: set[str] = set()
        self._drift_alerts: dict[str, int] = {}

    # ── Version CRUD ──────────────────────────────────────────

    def register(self, version: ModelVersion) -> ModelVersion:
        self._versions[version.id] = version
        logger.info("Registered model %s v%s (%s)", version.model_name, version.version, version.action_class)
        return version

    def get(self, version_id: UUID) -> ModelVersion | None:
        return self._versions.get(version_id)

    def list(
        self,
        action_class: ModelActionClass | None = None,
        status: ModelStatus | None = None,
        limit: int = 100,
    ) -> list[ModelVersion]:
        results = list(self._versions.values())
        if action_class:
            results = [v for v in results if v.action_class == action_class]
        if status:
            results = [v for v in results if v.status == status]
        results.sort(key=lambda v: v.created_at, reverse=True)
        return results[:limit]

    def update_status(self, version_id: UUID, status: ModelStatus) -> ModelVersion | None:
        v = self._versions.get(version_id)
        if v is None:
            return None
        v.status = status
        v.updated_at = datetime.now(timezone.utc)
        if status == ModelStatus.CHAMPION:
            key = v.action_class.value
            self._champions[key] = version_id
            if key in self._challengers and self._challengers[key] == version_id:
                del self._challengers[key]
        elif status == ModelStatus.CHALLENGER:
            self._challengers[v.action_class.value] = version_id
        return v

    def delete(self, version_id: UUID) -> bool:
        v = self._versions.pop(version_id, None)
        if v is None:
            return False
        key = v.action_class.value
        self._champions.pop(key, None)
        self._challengers.pop(key, None)
        return True

    # ── Champion / Challenger ─────────────────────────────────

    def get_champion(self, action_class: ModelActionClass) -> ModelVersion | None:
        vid = self._champions.get(action_class.value)
        return self._versions.get(vid) if vid else None

    def get_challenger(self, action_class: ModelActionClass) -> ModelVersion | None:
        vid = self._challengers.get(action_class.value)
        return self._versions.get(vid) if vid else None

    def promote_challenger(self, action_class: ModelActionClass) -> ModelVersion | None:
        challenger = self.get_challenger(action_class)
        if challenger is None:
            return None
        current_champ = self.get_champion(action_class)
        if current_champ:
            self.update_status(current_champ.id, ModelStatus.DEPRECATED)
        self.update_status(challenger.id, ModelStatus.CHAMPION)
        logger.info("Promoted %s v%s to champion for %s",
                     challenger.model_name, challenger.version, action_class)
        return challenger

    # ── Metadata ──────────────────────────────────────────────

    def get_metadata(self, model_name: str, action_class: ModelActionClass) -> ModelMetadata:
        all_versions = self.list(action_class=action_class)
        champ = self.get_champion(action_class)
        chal = self.get_challenger(action_class)
        total_preds = sum(self._prediction_counts.get(v.id, 0) for v in all_versions)
        avg_conf = 0.0
        total_conf_count = sum(self._confidence_count.get(v.id, 0) for v in all_versions)
        if total_conf_count > 0:
            avg_conf = sum(self._confidence_sum.get(v.id, 0.0) for v in all_versions) / total_conf_count
        key = action_class.value
        return ModelMetadata(
            model_name=model_name,
            action_class=action_class,
            current_champion=champ,
            current_challenger=chal,
            version_count=len(all_versions),
            last_trained_at=max((v.updated_at for v in all_versions), default=None),
            total_predictions=total_preds,
            avg_confidence=round(avg_conf, 4),
            fallback_active=key in self._fallback_active,
            drift_detected=self._drift_alerts.get(key, 0) > 0,
            needs_recertification=self._check_recertification(champ),
        )

    # ── Tracking ──────────────────────────────────────────────

    def record_prediction(self, version_id: UUID, confidence: float) -> None:
        self._prediction_counts[version_id] = self._prediction_counts.get(version_id, 0) + 1
        self._confidence_sum[version_id] = self._confidence_sum.get(version_id, 0.0) + confidence
        self._confidence_count[version_id] = self._confidence_count.get(version_id, 0) + 1

    def set_fallback(self, action_class: ModelActionClass, active: bool) -> None:
        key = action_class.value
        if active:
            self._fallback_active.add(key)
        else:
            self._fallback_active.discard(key)

    def record_drift(self, action_class: ModelActionClass) -> int:
        key = action_class.value
        count = self._drift_alerts.get(key, 0) + 1
        self._drift_alerts[key] = count
        return count

    def clear_drift(self, action_class: ModelActionClass) -> None:
        self._drift_alerts.pop(action_class.value, None)

    def clear_all(self) -> None:
        self._versions.clear()
        self._champions.clear()
        self._challengers.clear()
        self._prediction_counts.clear()
        self._confidence_sum.clear()
        self._confidence_count.clear()
        self._fallback_active.clear()
        self._drift_alerts.clear()

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _check_recertification(champ: ModelVersion | None) -> bool:
        if champ is None or champ.certified_until is None:
            return True
        return champ.certified_until < datetime.now(timezone.utc)


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
