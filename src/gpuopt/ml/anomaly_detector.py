from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpuopt.schemas import ClusterStateData

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    timestamp: datetime
    feature: str
    value: float
    score: float
    is_anomaly: bool
    severity: str
    explanation: str


@dataclass
class AnomalyReport:
    cluster_id: str
    generated_at: datetime
    scores: list[AnomalyScore] = field(default_factory=list)
    anomaly_count: int = 0
    total_metrics: int = 0

    @property
    def anomaly_ratio(self) -> float:
        return self.anomaly_count / max(self.total_metrics, 1)


class IsolationForest:
    def __init__(self, n_trees: int = 100, sample_size: int = 256, seed: int = 42) -> None:
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self._trees: list[_IsolationTree] = []
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        n = X.shape[0]
        sample_size = min(self.sample_size, n)
        rng = random.Random(self.seed)
        self._trees = []
        for i in range(self.n_trees):
            indices = rng.sample(range(n), sample_size)
            sample = X[indices]
            tree = _IsolationTree(seed=self.seed + i)
            tree.build(sample, 0, _max_depth(math.ceil(math.log2(sample_size))))
            self._trees.append(tree)
        self._fitted = True

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or not self._trees:
            return np.zeros(X.shape[0])
        depths = np.zeros((X.shape[0], len(self._trees)))
        for i, tree in enumerate(self._trees):
            depths[:, i] = tree.path_length(X)
        avg_depth = np.mean(depths, axis=1)
        n = self.sample_size
        c = _c_factor(n)
        scores = 2.0 ** (-avg_depth / c)
        return scores

    def predict(self, X: np.ndarray, threshold: float = 0.6) -> np.ndarray:
        scores = self.score_samples(X)
        return (scores > threshold).astype(int)


class _IsolationTree:
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._left: _IsolationTree | None = None
        self._right: _IsolationTree | None = None
        self._split_feature: int = 0
        self._split_value: float = 0.0
        self._size: int = 0
        self._external: bool = False

    def build(self, X: np.ndarray, depth: int, max_depth: int) -> None:
        self._size = X.shape[0]
        if depth >= max_depth or self._size <= 1:
            self._external = True
            return
        n_features = X.shape[1]
        self._split_feature = self._rng.randint(0, n_features - 1)
        col = X[:, self._split_feature]
        min_val, max_val = float(np.min(col)), float(np.max(col))
        if min_val >= max_val:
            self._external = True
            return
        self._split_value = self._rng.uniform(min_val, max_val)
        left_mask = col < self._split_value
        right_mask = ~left_mask
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            self._external = True
            return
        self._left = _IsolationTree(seed=self._rng.randint(0, 2**31))
        self._right = _IsolationTree(seed=self._rng.randint(0, 2**31))
        self._left.build(X[left_mask], depth + 1, max_depth)
        self._right.build(X[right_mask], depth + 1, max_depth)

    def path_length(self, X: np.ndarray) -> np.ndarray:
        if self._external:
            return np.full(X.shape[0], _c(self._size))
        col = X[:, self._split_feature]
        left_mask = col < self._split_value
        right_mask = ~left_mask
        depths = np.zeros(X.shape[0])
        if np.any(left_mask) and self._left is not None:
            depths[left_mask] = 1 + self._left.path_length(X[left_mask])
        if np.any(right_mask) and self._right is not None:
            depths[right_mask] = 1 + self._right.path_length(X[right_mask])
        return depths


def _c(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


def _c_factor(n: int) -> float:
    if n <= 1:
        return 1.0
    return _c(n)


def _max_depth(limit: int) -> int:
    return max(1, limit)


class TimeSeriesDecomposer:
    def __init__(self, period: int = 24) -> None:
        self.period = period

    def decompose(self, values: np.ndarray) -> dict[str, np.ndarray]:
        n = len(values)
        if n < self.period * 2:
            return {"trend": values, "seasonal": np.zeros(n), "residual": np.zeros(n)}
        trend = self._moving_average(values, self.period)
        detrended = values[:len(trend)] - trend
        seasonal = self._seasonal_component(detrended)
        residual = values[:len(seasonal)] - trend[:len(seasonal)] - seasonal
        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < window:
            return values
        cumsum = np.cumsum(np.insert(values, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    def _seasonal_component(self, values: np.ndarray) -> np.ndarray:
        n = len(values)
        seasonal = np.zeros(n)
        for i in range(self.period):
            indices = list(range(i, n, self.period))
            if indices:
                seasonal[indices] = np.mean(values[indices])
        seasonal -= np.mean(seasonal)
        return seasonal

    def detect_anomalies(self, values: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
        decomposed = self.decompose(values)
        residual = decomposed["residual"]
        if len(residual) < 2:
            return np.zeros(len(values))
        mean_r = np.mean(residual)
        std_r = np.std(residual)
        if std_r == 0:
            return np.zeros(len(values))
        z_scores = np.abs((residual - mean_r) / std_r)
        anomaly_mask = np.zeros(len(values), dtype=int)
        anomaly_mask[:len(z_scores)] = (z_scores > z_threshold).astype(int)
        return anomaly_mask


class MLAnomalyDetector:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path("./data/ml")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._isolation_forest: IsolationForest | None = None
        self._decomposer = TimeSeriesDecomposer(period=24)
        self._feature_history: dict[str, list[float]] = {}
        self._baseline_stats: dict[str, dict[str, float]] = {}
        self._load()

    def _state_path(self) -> Path:
        return self.model_dir / "anomaly_state.json"

    def _load(self) -> None:
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._feature_history = data.get("feature_history", {})
                self._baseline_stats = data.get("baseline_stats", {})
            except Exception as exc:
                logger.warning("Failed to load anomaly state: %s", exc)

    def _save(self) -> None:
        try:
            data = {
                "feature_history": self._feature_history,
                "baseline_stats": self._baseline_stats,
            }
            self._state_path().write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save anomaly state: %s", exc)

    def update(self, state: ClusterStateData) -> None:
        from gpuopt.ml.features import extract_state_features, extract_telemetry_features
        feats = extract_state_features(state)
        telemetry = getattr(state, "telemetry", None)
        if telemetry:
            try:
                feats.update(extract_telemetry_features(telemetry))
            except Exception:
                pass
        for key, val in feats.items():
            if key not in self._feature_history:
                self._feature_history[key] = []
            self._feature_history[key].append(float(val))
            max_history = 1000
            if len(self._feature_history[key]) > max_history:
                self._feature_history[key] = self._feature_history[key][-max_history:]

        if len(self._feature_history.get("memory_utilization_pct", [])) >= 10:
            self._fit_isolation_forest()

        self._update_baseline_stats(feats)
        self._save()

    def _fit_isolation_forest(self) -> None:
        keys = ["memory_utilization_pct", "gpu_count", "node_count", "pod_density",
                "avg_gpu_utilization", "avg_memory_utilization", "max_temperature"]
        rows = []
        n = min(len(self._feature_history.get(k, [])) for k in keys if k in self._feature_history)
        if n < 5:
            return
        for i in range(n):
            row = []
            for k in keys:
                hist = self._feature_history.get(k, [])
                row.append(hist[-(n - i)] if len(hist) >= n - i else 0.0)
            rows.append(row)
        X = np.array(rows, dtype=np.float64)
        if X.shape[0] >= 5:
            forest = IsolationForest(n_trees=50, sample_size=min(256, X.shape[0]))
            forest.fit(X)
            self._isolation_forest = forest

    def _update_baseline_stats(self, feats: dict[str, float]) -> None:
        for key, val in feats.items():
            if key not in self._baseline_stats:
                self._baseline_stats[key] = {"mean": val, "std": 0.0, "count": 1}
            else:
                stats = self._baseline_stats[key]
                old_mean = stats["mean"]
                stats["count"] += 1
                stats["mean"] = old_mean + (val - old_mean) / stats["count"]
                if stats["count"] > 1:
                    stats["std"] = math.sqrt(
                        (stats["std"] ** 2 * (stats["count"] - 2) + (val - old_mean) * (val - stats["mean"]))
                        / (stats["count"] - 1)
                    )

    def analyze_state(self, state: ClusterStateData) -> AnomalyReport:
        from gpuopt.ml.features import extract_state_features, extract_telemetry_features
        feats = extract_state_features(state)
        telemetry = getattr(state, "telemetry", None)
        if telemetry:
            try:
                feats.update(extract_telemetry_features(telemetry))
            except Exception:
                pass

        scores: list[AnomalyScore] = []
        now = datetime.now(timezone.utc)

        if self._isolation_forest is not None:
            keys = ["memory_utilization_pct", "gpu_count", "node_count", "pod_density",
                    "avg_gpu_utilization", "avg_memory_utilization", "max_temperature"]
            row = [feats.get(k, 0.0) for k in keys]
            X = np.array([row], dtype=np.float64)
            iforest_scores = self._isolation_forest.score_samples(X)
            if len(iforest_scores) > 0:
                for idx, key in enumerate(keys):
                    s = float(iforest_scores[0])
                    scores.append(AnomalyScore(
                        timestamp=now, feature=f"iforest_{key}",
                        value=feats.get(key, 0.0), score=s,
                        is_anomaly=s > 0.6,
                        severity="high" if s > 0.7 else ("medium" if s > 0.6 else "low"),
                        explanation=f"Isolation Forest anomaly score={s:.3f} for {key}={feats.get(key, 0.0):.2f}",
                    ))

        for key, val in feats.items():
            stats = self._baseline_stats.get(key)
            if stats is None or stats["count"] < 3:
                continue
            if stats["std"] < 1e-10:
                continue
            z_score = abs(val - stats["mean"]) / stats["std"]
            if z_score > 2.0:
                severity = "critical" if z_score > 4.0 else ("high" if z_score > 3.0 else "medium")
                scores.append(AnomalyScore(
                    timestamp=now, feature=key, value=val,
                    score=min(z_score / 5.0, 1.0), is_anomaly=True,
                    severity=severity,
                    explanation=f"Z-score={z_score:.2f} (mean={stats['mean']:.2f}, std={stats['std']:.2f}) for {key}={val:.2f}",
                ))

            hist = self._feature_history.get(key, [])
            if len(hist) >= 48:
                arr = np.array(hist[-48:])
                anomaly_mask = self._decomposer.detect_anomalies(arr, z_threshold=3.0)
                if len(anomaly_mask) > 0 and anomaly_mask[-1] == 1:
                    existing = any(s.feature == f"ts_{key}" for s in scores)
                    if not existing:
                        scores.append(AnomalyScore(
                            timestamp=now, feature=f"ts_{key}", value=val,
                            score=0.7, is_anomaly=True,
                            severity="medium",
                            explanation=f"Time-series anomaly detected for {key}={val:.2f}",
                        ))

        anomaly_count = sum(1 for s in scores if s.is_anomaly)
        report = AnomalyReport(
            cluster_id=str(state.cluster_id),
            generated_at=now,
            scores=scores,
            anomaly_count=anomaly_count,
            total_metrics=len(scores),
        )
        return report

    def reset(self) -> None:
        self._isolation_forest = None
        self._feature_history.clear()
        self._baseline_stats.clear()
        path = self._state_path()
        if path.exists():
            path.unlink()
        logger.info("MLAnomalyDetector reset")
