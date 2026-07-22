from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .models import (
    DriftReport,
    FallbackConfig,
    ModelActionClass,
    ModelVersion,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class FallbackEngine:
    def __init__(self, registry: ModelRegistry, config: FallbackConfig | None = None) -> None:
        self._registry = registry
        self._config = config or FallbackConfig()
        self._fallback_start: dict[str, datetime] = {}

    def should_fallback(
        self,
        version: ModelVersion,
        confidence: float,
        drift_reports: list[DriftReport],
    ) -> tuple[bool, str]:
        key = version.action_class.value

        if not self._config.enabled:
            return False, "Fallback disabled"

        if confidence < self._config.confidence_threshold:
            return True, f"Confidence {confidence:.2f} below threshold {self._config.confidence_threshold}"

        high_drift = [r for r in drift_reports if r.drift_score > self._config.drift_score_threshold]
        if high_drift:
            reasons = [f"{r.drift_type.value}={r.drift_score:.2f}" for r in high_drift]
            return True, f"Drift threshold exceeded: {', '.join(reasons)}"

        if version.training_data_window_end:
            age_hours = (datetime.now(timezone.utc) - version.training_data_window_end).total_seconds() / 3600
            if age_hours > self._config.max_data_stale_hours:
                return True, f"Training data {age_hours:.0f}h old (max {self._config.max_data_stale_hours}h)"

        # Check if currently in fallback and if recovery is possible
        if key in self._fallback_start:
            elapsed = (datetime.now(timezone.utc) - self._fallback_start[key]).total_seconds() / 3600
            if elapsed > self._config.max_fallback_duration_hours:
                logger.warning("Fallback for %s exceeded max duration %dh — forcing recovery check",
                               key, self._config.max_fallback_duration_hours)
                return False, "Forcing recovery check after max fallback duration"

        return False, ""

    def get_heuristic(self, version: ModelVersion, input_data: dict[str, Any]) -> dict[str, Any]:
        strategy = self._config.fallback_strategy
        if strategy == "deterministic_heuristic":
            return self._deterministic_heuristic(version, input_data)
        return {"fallback": True, "strategy": strategy, "input": input_data}

    def activate_fallback(self, action_class: ModelActionClass, reason: str) -> None:
        key = action_class.value
        self._fallback_start.setdefault(key, datetime.now(timezone.utc))
        self._registry.set_fallback(action_class, True)
        logger.warning("Fallback activated for %s: %s", key, reason)

    def deactivate_fallback(self, action_class: ModelActionClass) -> None:
        key = action_class.value
        self._fallback_start.pop(key, None)
        self._registry.set_fallback(action_class, False)
        logger.info("Fallback deactivated for %s", key)

    def is_fallback_active(self, action_class: ModelActionClass) -> bool:
        return action_class.value in self._fallback_start

    def check_recovery(
        self,
        version: ModelVersion,
        confidence: float,
        drift_reports: list[DriftReport],
    ) -> tuple[bool, str]:
        if not self._registry.get_champion(version.action_class):
            return False, "No champion model available for recovery"
        should, reason = self.should_fallback(version, confidence, drift_reports)
        if not should:
            self.deactivate_fallback(version.action_class)
            return True, "Recovered — confidence and drift within limits"
        return False, f"Still in fallback: {reason}"

    @staticmethod
    def _deterministic_heuristic(version: ModelVersion, data: dict[str, Any]) -> dict[str, Any]:
        action = version.action_class
        if action == ModelActionClass.RECOMMENDATION_SCORING:
            return {
                "fallback": True,
                "strategy": "deterministic_heuristic",
                "score": data.get("heuristic_score", 50.0),
                "confidence": 0.5,
            }
        if action == ModelActionClass.DEMAND_FORECAST:
            return {
                "fallback": True,
                "strategy": "deterministic_heuristic",
                "forecast_type": "linear_trend",
                "prediction": data.get("last_observed_value", 0.0),
                "confidence": 0.3,
            }
        if action == ModelActionClass.ANOMALY_DETECTION:
            return {
                "fallback": True,
                "strategy": "deterministic_heuristic",
                "anomaly_detected": False,
                "confidence": 0.4,
                "method": "z_score_only",
            }
        return {
            "fallback": True,
            "strategy": "deterministic_heuristic",
            "confidence": 0.3,
        }
