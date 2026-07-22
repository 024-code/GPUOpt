from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from gpuopt.ml.features import extract_rec_features, extract_state_features, extract_analysis_features
from gpuopt.schemas import (
    ClusterStateData,
    RecommendationSet,
    RecommendationStatus,
    ResourceRecommendation,
    WorkloadAnalysisResult,
)

logger = logging.getLogger(__name__)

_FEATURE_KEYS = [
    "severity_encoded", "type_encoded", "confidence", "risk_level_encoded",
    "total_estimated_savings", "action_count", "affected_resource_count", "current_score",
]

_STATE_FEATURE_KEYS = [
    "node_count", "gpu_count", "memory_utilization_pct", "memory_utilization_std",
    "gpu_idle_count", "gpu_hot_count", "node_ready_ratio", "memory_fragmentation",
]

_ANALYSIS_FEATURE_KEYS = [
    "overall_efficiency_score", "total_idle_gpu_hours", "estimated_power_waste_kwh",
    "avg_gpu_idle_pct", "avg_memory_pressure_pct",
]


class _NumpyRegressor:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.lr = learning_rate
        self._fitted = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_features = X.shape[1]
        if self.weights is None:
            self.weights = np.zeros(n_features)
        pred = X @ self.weights + self.bias
        error = y[0] - pred[0]
        self.weights += self.lr * error * X[0]
        self.bias += self.lr * error
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.full(X.shape[0], 50.0)
        return X @ self.weights + self.bias

    @property
    def coef_(self) -> np.ndarray:
        return self.weights if self.weights is not None else np.array([])


class _NumpyScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return X
        return (X - self.mean_) / self.scale_


class RecommendationModel:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path("./data/ml")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._regressor: _NumpyRegressor | None = None
        self._scaler: _NumpyScaler | None = None
        self._feature_importance: dict[str, float] = {}
        self._training_count: int = 0
        self._load_or_init()

    def _model_path(self) -> Path:
        return self.model_dir / "rec_model.pkl"

    def _scaler_path(self) -> Path:
        return self.model_dir / "rec_scaler.pkl"

    def _metadata_path(self) -> Path:
        return self.model_dir / "rec_metadata.json"

    def _load_or_init(self) -> None:
        model_path = self._model_path()
        scaler_path = self._scaler_path()
        metadata_path = self._metadata_path()
        if model_path.exists() and scaler_path.exists():
            try:
                with open(model_path, "rb") as f:
                    self._regressor = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                if metadata_path.exists():
                    meta = json.loads(metadata_path.read_text())
                    self._training_count = meta.get("training_count", 0)
                    self._feature_importance = meta.get("feature_importance", {})
                logger.info("Loaded recommendation model (%d training samples)", self._training_count)
            except Exception as exc:
                logger.warning("Failed to load model, initializing new: %s", exc)
                self._init_fresh()
        else:
            self._init_fresh()

    def _init_fresh(self) -> None:
        self._regressor = _NumpyRegressor(learning_rate=0.01)
        self._scaler = _NumpyScaler()
        self._training_count = 0
        self._feature_importance = {}
        dummy = np.zeros((2, len(_FEATURE_KEYS)))
        self._scaler.fit(dummy)
        self._regressor.partial_fit(self._scaler.transform(dummy[:1]), np.array([50.0]))

    def _build_feature_vector(
        self,
        rec: ResourceRecommendation,
        state: ClusterStateData | None = None,
        analysis: WorkloadAnalysisResult | None = None,
    ) -> np.ndarray:
        rec_feats = extract_rec_features(rec)
        vec = [rec_feats.get(k, 0.0) for k in _FEATURE_KEYS]
        if state:
            state_feats = extract_state_features(state)
            vec.extend([state_feats.get(k, 0.0) for k in _STATE_FEATURE_KEYS])
        else:
            vec.extend([0.0] * len(_STATE_FEATURE_KEYS))
        if analysis:
            analysis_feats = extract_analysis_features(analysis)
            vec.extend([analysis_feats.get(k, 0.0) for k in _ANALYSIS_FEATURE_KEYS])
        else:
            vec.extend([0.0] * len(_ANALYSIS_FEATURE_KEYS))
        return np.array(vec, dtype=np.float64).reshape(1, -1)

    def score_recommendation(
        self,
        rec: ResourceRecommendation,
        state: ClusterStateData | None = None,
        analysis: WorkloadAnalysisResult | None = None,
    ) -> float:
        if self._regressor is None or self._scaler is None:
            return self._numpy_score(rec)
        vec = self._build_feature_vector(rec, state, analysis)
        try:
            scaled = self._scaler.transform(vec)
            predicted = float(self._regressor.predict(scaled)[0])
            predicted = max(0.0, min(100.0, predicted))
            if self._training_count < 5:
                blend = self._training_count / 5.0
                return round(predicted * blend + rec.score * (1 - blend), 1)
            return round(predicted, 1)
        except Exception as exc:
            logger.debug("ML score fallback to heuristic: %s", exc)
            return rec.score

    def _numpy_score(self, rec: ResourceRecommendation) -> float:
        severity_map = {"critical": 90, "high": 70, "medium": 50, "low": 30, "info": 10}
        base = severity_map.get(rec.severity.value, 30)
        confidence_contrib = rec.confidence * 30
        risk_map = {"high": 15, "medium": 8, "low": 0}
        risk_contrib = risk_map.get(rec.risk_level, 0)
        savings_total = sum(abs(v) for v in rec.estimated_savings.values() if isinstance(v, (int, float)))
        savings_contrib = min(savings_total / 10.0, 15)
        score = min(base + confidence_contrib + risk_contrib + savings_contrib, 100)
        return round(score, 1)

    def score_recommendation_set(
        self,
        rec_set: RecommendationSet,
        state: ClusterStateData | None = None,
        analysis: WorkloadAnalysisResult | None = None,
    ) -> list[ResourceRecommendation]:
        scored: list[ResourceRecommendation] = []
        for rec in rec_set.recommendations:
            ml_score = self.score_recommendation(rec, state, analysis)
            rec.score = ml_score
            scored.append(rec)
        scored.sort(key=lambda r: (-r.score, -r.confidence))
        return scored

    def train_from_feedback(
        self,
        rec: ResourceRecommendation,
        outcome_score: float,
        state: ClusterStateData | None = None,
        analysis: WorkloadAnalysisResult | None = None,
    ) -> None:
        if self._regressor is None or self._scaler is None:
            self._training_count += 1
            logger.info("Feedback recorded (count=%d, score=%.1f)", self._training_count, outcome_score)
            return
        self._training_count += 1
        vec = self._build_feature_vector(rec, state, analysis)
        try:
            scaled = self._scaler.transform(vec)
            self._regressor.partial_fit(scaled, np.array([outcome_score]))
            self._update_feature_importance(scaled[0], outcome_score)
            self._save()
            logger.info("Model updated with feedback (count=%d, score=%.1f)", self._training_count, outcome_score)
        except Exception as exc:
            logger.warning("Failed to update model: %s", exc)

    def train_from_status(
        self,
        rec: ResourceRecommendation,
        old_status: str,
        new_status: str,
        state: ClusterStateData | None = None,
        analysis: WorkloadAnalysisResult | None = None,
    ) -> None:
        outcome_map = {
            "implemented": 95.0,
            "approved": 75.0,
            "pending": 50.0,
            "dismissed": 15.0,
        }
        if new_status in outcome_map and old_status != new_status:
            self.train_from_feedback(rec, outcome_map[new_status], state, analysis)

    def train_batch(
        self,
        recs: list[tuple[ResourceRecommendation, float]],
    ) -> None:
        if not recs or self._regressor is None or self._scaler is None:
            if recs:
                self._training_count += len(recs)
            return
        X: list[np.ndarray] = []
        y: list[float] = []
        for rec, score in recs:
            vec = self._build_feature_vector(rec)
            X.append(vec[0])
            y.append(score)
        try:
            X_arr = np.array(X)
            y_arr = np.array(y)
            if len(X_arr) > 1:
                scaler = _NumpyScaler()
                X_scaled = scaler.fit_transform(X_arr)
                reg = _NumpyRegressor(learning_rate=0.01)
                for i in range(len(X_scaled)):
                    reg.partial_fit(X_scaled[i:i+1], y_arr[i:i+1])
                self._regressor = reg
                self._scaler = scaler
                self._training_count += len(recs)
                self._save()
                logger.info("Batch trained on %d samples (total=%d)", len(recs), self._training_count)
            else:
                self.train_from_feedback(recs[0][0], recs[0][1])
        except Exception as exc:
            logger.warning("Batch training error: %s", exc)

    def _update_feature_importance(self, feature_vector: np.ndarray, target: float) -> None:
        if self._regressor is None:
            return
        coefs = self._regressor.coef_
        all_keys = _FEATURE_KEYS + _STATE_FEATURE_KEYS + _ANALYSIS_FEATURE_KEYS
        if len(coefs) == 0:
            return
        total_imp = sum(abs(c) for c in coefs) or 1.0
        for i, key in enumerate(all_keys):
            if i < len(coefs):
                self._feature_importance[key] = round(abs(coefs[i]) / total_imp * 100, 2)

    def get_feature_importance(self) -> dict[str, float]:
        return dict(self._feature_importance)

    def get_training_count(self) -> int:
        return self._training_count

    def _save(self) -> None:
        if self._regressor is None or self._scaler is None:
            return
        try:
            with open(self._model_path(), "wb") as f:
                pickle.dump(self._regressor, f)
            with open(self._scaler_path(), "wb") as f:
                pickle.dump(self._scaler, f)
            meta = {
                "training_count": self._training_count,
                "feature_importance": self._feature_importance,
            }
            self._metadata_path().write_text(json.dumps(meta, indent=2))
        except Exception as exc:
            logger.warning("Failed to save model: %s", exc)

    def reset(self) -> None:
        self._init_fresh()
        for p in [self._model_path(), self._scaler_path(), self._metadata_path()]:
            if p.exists():
                p.unlink()
        logger.info("Recommendation model reset to initial state")
