from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .approval import ApprovalManager
from .champion_challenger import ChampionChallenger
from .drift_monitor import DriftMonitor
from .fallback import FallbackEngine
from .models import (
    DriftReport,
    GovernanceConfig,
    ModelActionClass,
    ModelStatus,
    ModelVersion,
    ShadowEvaluation,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelGovernor:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        config: GovernanceConfig | None = None,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._config = config or GovernanceConfig()
        self._champion_challenger = ChampionChallenger(self._registry, self._config.champion_challenger)
        self._drift_monitor = DriftMonitor(self._registry)
        self._fallback = FallbackEngine(self._registry, self._config.fallback)
        self._approval = ApprovalManager(self._registry, self._config)

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def champion_challenger(self) -> ChampionChallenger:
        return self._champion_challenger

    @property
    def drift_monitor(self) -> DriftMonitor:
        return self._drift_monitor

    @property
    def fallback(self) -> FallbackEngine:
        return self._fallback

    @property
    def approval(self) -> ApprovalManager:
        return self._approval

    # ── Registration ──────────────────────────────────────────

    def register_model(
        self,
        model_name: str,
        version: str,
        action_class: ModelActionClass,
        owner: str = "",
        features: list[str] | None = None,
        training_metrics: dict[str, float] | None = None,
        training_window: tuple[datetime, datetime] | None = None,
    ) -> ModelVersion:
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            action_class=action_class,
            owner=owner,
            features=features or [],
            training_metrics=training_metrics or {},
        )
        if training_window:
            model_version.training_data_window_start = training_window[0]
            model_version.training_data_window_end = training_window[1]
        self._registry.register(model_version)

        if self._approval.requires_approval(model_version):
            self._approval.request_approval(model_version, requested_by=owner)
            logger.info("High-impact model %s v%s requires approval", model_name, version)
        else:
            self._registry.update_status(model_version.id, ModelStatus.CHALLENGER)

        return model_version

    def set_champion(self, version_id: UUID) -> ModelVersion | None:
        version = self._registry.get(version_id)
        if version is None:
            return None
        if self._approval.requires_approval(version) and version.approved_by is None:
            logger.warning("Cannot promote %s v%s to champion — approval required", version.model_name, version.version)
            return None
        if version.certified_until and version.certified_until < datetime.now(timezone.utc):
            logger.warning("Cannot promote %s v%s to champion — certification expired", version.model_name, version.version)
            return None
        return self._registry.promote_challenger(version.action_class)

    # ── Prediction lifecycle ──────────────────────────────────

    def predict(
        self,
        action_class: ModelActionClass,
        input_data: dict[str, Any],
        confidence: float = 0.8,
    ) -> dict[str, Any]:
        champion = self._registry.get_champion(action_class)
        if champion is None:
            return {"fallback": True, "reason": "No champion model registered", "result": None}

        drift_reports = self._drift_monitor.check_all(champion)
        should_fallback, fallback_reason = self._fallback.should_fallback(
            champion, confidence, drift_reports,
        )

        if should_fallback:
            self._fallback.activate_fallback(action_class, fallback_reason)
            heuristic = self._fallback.get_heuristic(champion, input_data)
            return {
                "fallback": True,
                "reason": fallback_reason,
                "drift_reports": [r.model_dump(mode="json") for r in drift_reports if r.drift_score > 0],
                "result": heuristic,
            }

        if self._fallback.is_fallback_active(action_class):
            recovered, msg = self._fallback.check_recovery(champion, confidence, drift_reports)
            if not recovered:
                heuristic = self._fallback.get_heuristic(champion, input_data)
                return {"fallback": True, "reason": msg, "result": heuristic}

        self._registry.record_prediction(champion.id, confidence)
        return {
            "fallback": False,
            "model": champion.model_name,
            "version": champion.version,
            "result": None,
            "confidence": confidence,
        }

    def record_observation(
        self,
        action_class: ModelActionClass,
        features: dict[str, float] | None = None,
        prediction_error: float | None = None,
        outcome_divergence: float | None = None,
        workload_distribution: dict[str, int] | None = None,
    ) -> None:
        champion = self._registry.get_champion(action_class)
        if champion is None:
            return
        if features:
            self._drift_monitor.record_features(champion, features)
        if prediction_error is not None:
            self._drift_monitor.record_prediction_error(champion, prediction_error)
        if outcome_divergence is not None:
            self._drift_monitor.record_outcome(champion, None, None, divergence=outcome_divergence)
        if workload_distribution:
            self._drift_monitor.record_workload_distribution(champion, workload_distribution)

    # ── Periodic maintenance ──────────────────────────────────

    def run_periodic_checks(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "recertification_due": [],
            "drift_alerts": [],
            "fallback_status": {},
            "champion_challenger_evals": [],
        }

        for action_class in ModelActionClass:
            champion = self._registry.get_champion(action_class)
            if champion:
                drift_reports = self._drift_monitor.check_all(champion)
                significant = [r for r in drift_reports if r.drift_score > 0.3]
                results["drift_alerts"].extend(
                    {"action_class": action_class.value, "report": r.model_dump(mode="json")}
                    for r in significant
                )

                score = 0.8
                should_fb, reason = self._fallback.should_fallback(champion, score, drift_reports)
                results["fallback_status"][action_class.value] = {
                    "active": self._fallback.is_fallback_active(action_class),
                    "should_fallback": should_fb,
                    "reason": reason if should_fb else "",
                }

        recert = self._approval.check_recertification()
        results["recertification_due"] = [
            {"request_id": str(r.id), "model_version_id": str(r.model_version_id),
             "status": r.status.value, "certified_until": r.certified_until.isoformat() if r.certified_until else None}
            for r in recert
        ]

        results["fallback_status"]["_config"] = self._config.fallback.model_dump(mode="json")
        return results

    def get_champion_challenger(self) -> ChampionChallenger:
        return self._champion_challenger


_governor = ModelGovernor()


def get_governor() -> ModelGovernor:
    return _governor
