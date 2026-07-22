from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from gpuopt.ml.features import build_time_series_matrix, extract_state_features, extract_telemetry_features
from gpuopt.schemas import ClusterStateData, DemandForecastPoint

logger = logging.getLogger(__name__)

_MIN_SAMPLES_FOR_ML = 3


class ForecastModel:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path("./data/ml")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, Any] = {}
        self._feature_history: dict[str, list[tuple[datetime, dict[str, float]]]] = {}
        self._training_count: int = 0
        self._load()

    def _model_path(self) -> Path:
        return self.model_dir / "forecast_model.pkl"

    def _history_path(self) -> Path:
        return self.model_dir / "forecast_history.json"

    def _load(self) -> None:
        model_path = self._model_path()
        history_path = self._history_path()
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                self._models = data.get("models", {})
                self._training_count = data.get("training_count", 0)
                logger.info("Loaded forecast model (%d training samples)", self._training_count)
            except Exception as exc:
                logger.warning("Failed to load forecast model: %s", exc)
        if history_path.exists():
            try:
                raw = json.loads(history_path.read_text())
                self._feature_history = {
                    k: [(datetime.fromisoformat(t), v) for t, v in entries]
                    for k, v in raw.items()
                }
            except Exception as exc:
                logger.warning("Failed to load forecast history: %s", exc)

    def _save(self) -> None:
        try:
            data = {"models": self._models, "training_count": self._training_count}
            with open(self._model_path(), "wb") as f:
                pickle.dump(data, f)
            raw = {
                k: [(t.isoformat(), v) for t, v in entries]
                for k, v in self._feature_history.items()
            }
            self._history_path().write_text(json.dumps(raw, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save forecast model: %s", exc)

    def update_history(self, state: ClusterStateData) -> None:
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
            self._feature_history[key].append((state.collected_at, val))
            max_history = 1000
            if len(self._feature_history[key]) > max_history:
                self._feature_history[key] = self._feature_history[key][-max_history:]

    def _get_series(self, feature_key: str) -> tuple[np.ndarray, np.ndarray]:
        entries = self._feature_history.get(feature_key, [])
        if len(entries) < 2:
            return np.array([]), np.array([])
        t0 = entries[0][0].timestamp()
        times = np.array([(e[0].timestamp() - t0) / 3600.0 for e in entries])
        values = np.array([e[1] for e in entries])
        return times, values

    def forecast_feature(
        self,
        feature_key: str,
        horizon_hours: int = 24,
        steps: int = 24,
    ) -> list[tuple[datetime, float, float, float]]:
        times, values = self._get_series(feature_key)
        n = len(values)

        if n < 2:
            return []

        now = datetime.now(timezone.utc)
        results: list[tuple[datetime, float, float, float]] = []

        if n < _MIN_SAMPLES_FOR_ML:
            trend = (values[-1] - values[0]) / max(times[-1] - times[0], 0.01)
            for i in range(1, steps + 1):
                t = now + timedelta(hours=i * horizon_hours / steps)
                pred = values[-1] + trend * (i * horizon_hours / steps)
                noise_std = max(np.std(values) * 0.3, 0.01)
                lower = pred - 1.96 * noise_std
                upper = pred + 1.96 * noise_std
                results.append((t, float(pred), float(max(lower, 0)), float(upper)))
            return results

        try:
            X = times.reshape(-1, 1)
            y = values

            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-10
            X_scaled = (X - X_mean) / X_std

            degree = min(3, n - 1)
            X_poly = np.column_stack([X_scaled ** d for d in range(1, degree + 1)])

            X_design = np.column_stack([np.ones(n), X_poly])
            theta = np.linalg.lstsq(X_design, y, rcond=None)[0]

            residuals = y - X_design @ theta
            residual_std = max(np.std(residuals), 0.01)

            model_key = f"poly_{feature_key}"
            self._models[model_key] = {
                "theta": theta.tolist(),
                "degree": degree,
                "x_mean": float(X_mean),
                "x_std": float(X_std),
            }

            last_t = times[-1]
            for i in range(1, steps + 1):
                t = now + timedelta(hours=i * horizon_hours / steps)
                future_t = last_t + i * horizon_hours / steps
                f_scaled = (future_t - X_mean) / X_std
                f_poly = np.array([f_scaled ** d for d in range(1, degree + 1)])
                f_design = np.insert(f_poly, 0, 1.0)
                pred = float(f_design @ theta)
                ci = float(scipy_stats.norm.ppf(0.975)) * residual_std * np.sqrt(1 + 1.0 / n)
                lower = pred - ci
                upper = pred + ci
                results.append((t, float(pred), float(max(lower, 0)), float(upper)))

        except Exception as exc:
            logger.debug("ML forecast failed for %s, using trend: %s", feature_key, exc)
            if n >= 2:
                trend = (values[-1] - values[0]) / max(times[-1] - times[0], 0.01)
                for i in range(1, steps + 1):
                    t = now + timedelta(hours=i * horizon_hours / steps)
                    pred = values[-1] + trend * (i * horizon_hours / steps)
                    noise_std = max(np.std(values) * 0.3, 0.01)
                    lower = pred - 1.96 * noise_std
                    upper = pred + 1.96 * noise_std
                    results.append((t, float(pred), float(max(lower, 0)), float(upper)))

        return results

    def forecast_gpu_utilization(
        self,
        horizon_hours: int = 24,
        steps: int = 24,
    ) -> list[DemandForecastPoint]:
        util_forecast = self.forecast_feature("memory_utilization_pct", horizon_hours, steps)
        mem_forecast = self.forecast_feature("used_memory_gb", horizon_hours, steps)
        pod_forecast = self.forecast_feature("pod_density", horizon_hours, steps)

        points: list[DemandForecastPoint] = []
        for i in range(steps):
            util = util_forecast[i] if i < len(util_forecast) else None
            mem = mem_forecast[i] if i < len(mem_forecast) else None
            pod = pod_forecast[i] if i < len(pod_forecast) else None
            ts = now = datetime.now(timezone.utc) + timedelta(hours=(i + 1) * horizon_hours / steps)
            if util:
                ts = util[0]
            points.append(DemandForecastPoint(
                timestamp=ts,
                predicted_gpu_utilization_percent=round(util[1], 1) if util else 50.0,
                predicted_gpu_memory_used_bytes=round(mem[1] * (1024**3), 1) if mem else 0.0,
                predicted_pod_count=round(pod[1] * 110, 1) if pod else 0.0,
                confidence_lower=round(util[2], 1) if util else 0.0,
                confidence_upper=round(util[3], 1) if util else 100.0,
            ))
        return points

    def predict_idle_gpus(self, total_gpus: int) -> float:
        util_forecast = self.forecast_feature("memory_utilization_pct", horizon_hours=24, steps=24)
        if not util_forecast:
            return total_gpus * 0.4
        avg_predicted_util = np.mean([u[1] for u in util_forecast])
        return total_gpus * (1 - min(avg_predicted_util / 100, 1.0))

    def predict_peak_memory(self) -> float:
        mem_forecast = self.forecast_feature("used_memory_gb", horizon_hours=24, steps=24)
        if not mem_forecast:
            return 0.0
        peak = max(m[3] for m in mem_forecast)
        return float(peak)

    def predict_avg_utilization(self) -> float:
        util_forecast = self.forecast_feature("avg_gpu_utilization", horizon_hours=24, steps=24)
        if not util_forecast:
            return 50.0
        return float(np.mean([u[1] for u in util_forecast]))

    def get_training_count(self) -> int:
        return self._training_count

    def reset(self) -> None:
        self._models.clear()
        self._feature_history.clear()
        self._training_count = 0
        for p in [self._model_path(), self._history_path()]:
            if p.exists():
                p.unlink()
        logger.info("Forecast model reset")
