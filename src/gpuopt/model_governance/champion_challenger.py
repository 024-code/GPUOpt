from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .models import (
    ChampionChallengerConfig,
    ModelActionClass,
    ModelStatus,
    ModelVersion,
    ShadowEvaluation,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class ChampionChallenger:
    def __init__(self, registry: ModelRegistry, config: ChampionChallengerConfig | None = None) -> None:
        self._registry = registry
        self._config = config or ChampionChallengerConfig()
        self._evaluations: dict[UUID, ShadowEvaluation] = {}

    def start_evaluation(
        self,
        champion: ModelVersion,
        challenger: ModelVersion,
    ) -> ShadowEvaluation:
        if champion.status != ModelStatus.CHAMPION:
            self._registry.update_status(champion.id, ModelStatus.CHAMPION)
        if challenger.status != ModelStatus.CHALLENGER:
            self._registry.update_status(challenger.id, ModelStatus.CHALLENGER)

        evaluation = ShadowEvaluation(
            champion_id=champion.id,
            challenger_id=challenger.id,
            action_class=champion.action_class,
        )
        self._evaluations[evaluation.id] = evaluation
        challenger.shadow_evaluation_id = str(evaluation.id)
        logger.info("Started shadow evaluation %s: champion=%s v%s challenger=%s v%s",
                     evaluation.id, champion.model_name, champion.version,
                     challenger.model_name, challenger.version)
        return evaluation

    def record_result(
        self,
        evaluation_id: UUID,
        champion_score: float,
        challenger_score: float,
        metrics: dict[str, dict[str, float]] | None = None,
    ) -> ShadowEvaluation:
        evaluation = self._evaluations.get(evaluation_id)
        if evaluation is None:
            raise KeyError(f"Evaluation not found: {evaluation_id}")

        evaluation.sample_count += 1
        if champion_score > challenger_score:
            evaluation.champion_wins += 1
        elif challenger_score > champion_score:
            evaluation.challenger_wins += 1
        else:
            evaluation.ties += 1

        if metrics:
            for metric_name, values in metrics.items():
                if metric_name not in evaluation.metric_comparison:
                    evaluation.metric_comparison[metric_name] = {"champion_sum": 0.0, "challenger_sum": 0.0, "count": 0}
                evaluation.metric_comparison[metric_name]["champion_sum"] += values.get("champion", 0.0)
                evaluation.metric_comparison[metric_name]["challenger_sum"] += values.get("challenger", 0.0)
                evaluation.metric_comparison[metric_name]["count"] += 1

        return evaluation

    def complete_evaluation(self, evaluation_id: UUID) -> ShadowEvaluation:
        evaluation = self._evaluations.get(evaluation_id)
        if evaluation is None:
            raise KeyError(f"Evaluation not found: {evaluation_id}")

        evaluation.completed_at = datetime.now(timezone.utc)
        total = evaluation.sample_count
        if total == 0:
            evaluation.summary = "No samples collected"
            return evaluation

        champion_rate = evaluation.champion_wins / total
        challenger_rate = evaluation.challenger_wins / total

        for metric_name, data in evaluation.metric_comparison.items():
            if data["count"] > 0:
                data["champion_avg"] = round(data["champion_sum"] / data["count"], 4)
                data["challenger_avg"] = round(data["challenger_sum"] / data["count"], 4)
                data["delta"] = round(data["challenger_avg"] - data["champion_avg"], 4)

        auto_promote = self._config.auto_promote
        if challenger_rate >= self._config.win_threshold and total >= self._config.shadow_sample_minimum:
            if auto_promote:
                self._registry.promote_challenger(evaluation.action_class)
                evaluation.promoted = True
                evaluation.summary = (
                    f"Challenger won {challenger_rate:.1%} vs champion {champion_rate:.1%} "
                    f"over {total} samples — auto-promoted"
                )
            else:
                evaluation.summary = (
                    f"Challenger won {challenger_rate:.1%} vs champion {champion_rate:.1%} "
                    f"over {total} samples — ready for manual promotion"
                )
        elif champion_rate >= self._config.win_threshold:
            evaluation.rejected = True
            evaluation.summary = (
                f"Champion retained with {champion_rate:.1%} vs challenger {challenger_rate:.1%} "
                f"over {total} samples"
            )
        else:
            evaluation.summary = (
                f"Inconclusive: champion {champion_rate:.1%} challenger {challenger_rate:.1%} "
                f"over {total} samples (need {self._config.shadow_sample_minimum} min)"
            )

        self._registry.record_prediction(evaluation.champion_id, champion_rate)
        self._registry.record_prediction(evaluation.challenger_id, challenger_rate)
        logger.info("Completed shadow evaluation %s: %s", evaluation_id, evaluation.summary)
        return evaluation

    def get_evaluation(self, evaluation_id: UUID) -> ShadowEvaluation | None:
        return self._evaluations.get(evaluation_id)

    def list_evaluations(self, action_class: ModelActionClass | None = None) -> list[ShadowEvaluation]:
        results = list(self._evaluations.values())
        if action_class:
            results = [e for e in results if e.action_class == action_class]
        results.sort(key=lambda e: e.started_at, reverse=True)
        return results

    def evaluate_shadow(
        self,
        champion_prediction: Any,
        challenger_prediction: Any,
        actual_outcome: Any,
        evaluation_id: UUID,
    ) -> ShadowEvaluation:
        champion_score = self._score_prediction(champion_prediction, actual_outcome)
        challenger_score = self._score_prediction(challenger_prediction, actual_outcome)
        metrics = self._compute_metrics(champion_prediction, challenger_prediction, actual_outcome)
        return self.record_result(evaluation_id, champion_score, challenger_score, metrics)

    @staticmethod
    def _score_prediction(prediction: Any, actual: Any) -> float:
        if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
            if actual == 0:
                return 1.0 if prediction == 0 else 0.0
            error = abs(prediction - actual) / max(abs(actual), 1e-10)
            return max(0.0, 1.0 - error)
        return 1.0 if prediction == actual else 0.0

    @staticmethod
    def _compute_metrics(champion_pred: Any, challenger_pred: Any, actual: Any) -> dict[str, dict[str, float]]:
        champ_err = abs(champion_pred - actual) if isinstance(champion_pred, (int, float)) and isinstance(actual, (int, float)) else 0.0
        chal_err = abs(challenger_pred - actual) if isinstance(challenger_pred, (int, float)) and isinstance(actual, (int, float)) else 0.0
        return {
            "absolute_error": {"champion": champ_err, "challenger": chal_err},
        }
