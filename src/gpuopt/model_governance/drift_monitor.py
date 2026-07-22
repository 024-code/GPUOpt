from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .models import (
    DriftReport,
    DriftType,
    ModelActionClass,
    ModelVersion,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

MAX_HISTORY = 1000


class DriftMonitor:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._feature_baselines: dict[str, dict[str, float]] = {}
        self._feature_history: dict[str, deque[dict[str, float]]] = {}
        self._outcome_history: dict[str, deque[dict[str, Any]]] = {}
        self._prediction_errors: dict[str, deque[float]] = {}
        self._workload_distributions: dict[str, deque[dict[str, int]]] = {}
        self._reports: dict[UUID, DriftReport] = {}

    def check_all(self, version: ModelVersion) -> list[DriftReport]:
        reports: list[DriftReport] = []
        reports.append(self.check_feature_distribution(version))
        reports.append(self.check_prediction_error(version))
        reports.append(self.check_action_outcome_divergence(version))
        reports.append(self.check_workload_mix(version))
        reports.append(self.check_data_staleness(version))
        for r in reports:
            if r.drift_score > 0.3:
                self._reports[r.id] = r
                count = self._registry.record_drift(version.action_class)
                logger.warning("Drift %s for %s v%s (score=%.2f, alert #%d)",
                               r.drift_type, version.model_name, version.version, r.drift_score, count)
        return reports

    # ── Individual drift checks ───────────────────────────────

    def check_feature_distribution(self, version: ModelVersion) -> DriftReport:
        key = self._model_key(version)
        baseline = self._feature_baselines.get(key)
        history = self._feature_history.get(key, deque(maxlen=MAX_HISTORY))

        if baseline is None or not history:
            return DriftReport(
                model_version_id=version.id,
                drift_type=DriftType.FEATURE_DISTRIBUTION,
                drift_score=0.0,
                action_class=version.action_class,
                description="Insufficient data for feature distribution drift check",
            )

        latest = history[-1]
        drifted: list[str] = []
        deltas: dict[str, float] = {}
        max_drift = 0.0

        for feat, baseline_mean in baseline.items():
            if feat in latest:
                feat_val = latest[feat]
                if abs(baseline_mean) > 1e-10:
                    delta = abs(feat_val - baseline_mean) / abs(baseline_mean)
                else:
                    delta = abs(feat_val - baseline_mean)
                deltas[feat] = round(delta, 4)
                if delta > 0.2:
                    drifted.append(feat)
                    max_drift = max(max_drift, delta)

        score = min(max_drift, 1.0)
        severity = "low"
        if score > 0.5:
            severity = "medium"
        if score > 0.8:
            severity = "high"

        return DriftReport(
            model_version_id=version.id,
            drift_type=DriftType.FEATURE_DISTRIBUTION,
            drift_score=round(score, 4),
            drifted_features=drifted,
            feature_deltas=deltas,
            action_class=version.action_class,
            severity=severity,
            description=f"Feature distribution drift: {len(drifted)} drifted features (max delta={max_drift:.2f})",
            recommended_action="Retrain model with recent data" if score > 0.5 else "",
        )

    def check_prediction_error(self, version: ModelVersion) -> DriftReport:
        key = f"err_{self._model_key(version)}"
        errors = self._prediction_errors.get(key, deque(maxlen=MAX_HISTORY))
        if len(errors) < 10:
            return DriftReport(
                model_version_id=version.id,
                drift_type=DriftType.PREDICTION_ERROR,
                drift_score=0.0,
                action_class=version.action_class,
                description="Insufficient prediction error data",
            )

        recent = list(errors)[-50:]
        mean_err = sum(recent) / len(recent)
        baseline_err = sum(list(errors)[:50]) / min(50, len(errors))
        if baseline_err > 0:
            drift = (mean_err - baseline_err) / baseline_err
        else:
            drift = mean_err
        score = min(max(drift, 0.0), 1.0)

        severity = "low"
        if score > 0.3:
            severity = "medium"
        if score > 0.6:
            severity = "high"

        return DriftReport(
            model_version_id=version.id,
            drift_type=DriftType.PREDICTION_ERROR,
            drift_score=round(score, 4),
            action_class=version.action_class,
            severity=severity,
            description=f"Prediction error drift: baseline={baseline_err:.4f} recent={mean_err:.4f}",
            recommended_action="Investigate model accuracy degradation" if score > 0.3 else "",
        )

    def check_action_outcome_divergence(self, version: ModelVersion) -> DriftReport:
        key = f"out_{self._model_key(version)}"
        outcomes = self._outcome_history.get(key, deque(maxlen=MAX_HISTORY))
        if len(outcomes) < 5:
            return DriftReport(
                model_version_id=version.id,
                drift_type=DriftType.ACTION_OUTCOME_DIVERGENCE,
                drift_score=0.0,
                action_class=version.action_class,
                description="Insufficient outcome data",
            )

        recent = list(outcomes)[-20:]
        divergences = [o.get("divergence", 0.0) for o in recent if "divergence" in o]
        if not divergences:
            return DriftReport(
                model_version_id=version.id,
                drift_type=DriftType.ACTION_OUTCOME_DIVERGENCE,
                drift_score=0.0,
                action_class=version.action_class,
                description="No divergence data recorded",
            )
        avg_div = sum(divergences) / len(divergences)
        score = min(avg_div, 1.0)

        severity = "low"
        if score > 0.3:
            severity = "medium"
        if score > 0.6:
            severity = "high"

        return DriftReport(
            model_version_id=version.id,
            drift_type=DriftType.ACTION_OUTCOME_DIVERGENCE,
            drift_score=round(score, 4),
            action_class=version.action_class,
            severity=severity,
            description=f"Action-outcome divergence: avg divergence={avg_div:.2f} over {len(divergences)} actions",
            recommended_action="Review action effectiveness and update model if needed" if score > 0.3 else "",
        )

    def check_workload_mix(self, version: ModelVersion) -> DriftReport:
        key = f"wl_{self._model_key(version)}"
        distributions = self._workload_distributions.get(key, deque(maxlen=MAX_HISTORY))
        if len(distributions) < 2:
            return DriftReport(
                model_version_id=version.id,
                drift_type=DriftType.WORKLOAD_MIX,
                drift_score=0.0,
                action_class=version.action_class,
                description="Insufficient workload distribution data",
            )

        first = distributions[0]
        last = distributions[-1]
        total_drift = 0.0
        all_keys = set(first.keys()) | set(last.keys())
        for k in all_keys:
            v1 = first.get(k, 0)
            v2 = last.get(k, 0)
            total = v1 + v2
            if total > 0:
                total_drift += abs(v1 / total - v2 / total) if total > 0 else 0
        score = min(total_drift / max(len(all_keys), 1), 1.0)

        severity = "low"
        if score > 0.3:
            severity = "medium"
        if score > 0.6:
            severity = "high"

        return DriftReport(
            model_version_id=version.id,
            drift_type=DriftType.WORKLOAD_MIX,
            drift_score=round(score, 4),
            action_class=version.action_class,
            severity=severity,
            description=f"Workload mix drift: score={score:.2f}",
            recommended_action="Evaluate if training data still represents current workload" if score > 0.3 else "",
        )

    def check_data_staleness(self, version: ModelVersion) -> DriftReport:
        if version.training_data_window_end is None:
            return DriftReport(
                model_version_id=version.id,
                drift_type=DriftType.DATA_STALENESS,
                drift_score=0.0,
                action_class=version.action_class,
                description="No training window recorded",
            )

        now = datetime.now(timezone.utc)
        age_hours = (now - version.training_data_window_end).total_seconds() / 3600
        score = min(age_hours / 720, 1.0)  # 30 days = full drift

        severity = "low"
        if age_hours > 168:
            severity = "medium"
        if age_hours > 720:
            severity = "high"

        return DriftReport(
            model_version_id=version.id,
            drift_type=DriftType.DATA_STALENESS,
            drift_score=round(score, 4),
            action_class=version.action_class,
            severity=severity,
            description=f"Training data {age_hours:.0f}h old",
            recommended_action="Retrain with fresh data" if age_hours > 168 else "",
        )

    # ── Recording observations ────────────────────────────────

    def record_features(self, version: ModelVersion, features: dict[str, float]) -> None:
        key = self._model_key(version)
        if key not in self._feature_history:
            self._feature_history[key] = deque(maxlen=MAX_HISTORY)
            self._feature_baselines[key] = dict(features)
        self._feature_history[key].append(features)

    def record_prediction_error(self, version: ModelVersion, error: float) -> None:
        key = f"err_{self._model_key(version)}"
        if key not in self._prediction_errors:
            self._prediction_errors[key] = deque(maxlen=MAX_HISTORY)
        self._prediction_errors[key].append(error)

    def record_outcome(self, version: ModelVersion, predicted: Any, actual: Any, divergence: float = 0.0) -> None:
        key = f"out_{self._model_key(version)}"
        if key not in self._outcome_history:
            self._outcome_history[key] = deque(maxlen=MAX_HISTORY)
        self._outcome_history[key].append({
            "predicted": predicted,
            "actual": actual,
            "divergence": divergence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_workload_distribution(self, version: ModelVersion, distribution: dict[str, int]) -> None:
        key = f"wl_{self._model_key(version)}"
        if key not in self._workload_distributions:
            self._workload_distributions[key] = deque(maxlen=MAX_HISTORY)
        self._workload_distributions[key].append(distribution)

    # ── Query ─────────────────────────────────────────────────

    def list_reports(self, version_id: UUID | None = None, limit: int = 100) -> list[DriftReport]:
        results = list(self._reports.values())
        if version_id:
            results = [r for r in results if r.model_version_id == version_id]
        results.sort(key=lambda r: r.detected_at, reverse=True)
        return results[:limit]

    def clear_reports(self) -> None:
        self._reports.clear()

    def reset(self) -> None:
        self._feature_baselines.clear()
        self._feature_history.clear()
        self._outcome_history.clear()
        self._prediction_errors.clear()
        self._workload_distributions.clear()
        self._reports.clear()

    @staticmethod
    def _model_key(version: ModelVersion) -> str:
        return f"{version.model_name}/{version.version}"
