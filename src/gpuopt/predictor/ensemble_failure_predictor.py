from __future__ import annotations

import json
import logging
import pickle
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnsembleFailurePredictor:
    VERSION = "2.0.0"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        if model_dir is None:
            from ..config import get_settings
            settings = get_settings()
            model_dir = Path(settings.database_path).parent / "models"

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        self.nn = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), activation="relu",
            solver="adam", max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=42,
        )
        self.anomaly = IsolationForest(contamination=0.08, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()

        self.feature_names: list[str] = []
        self.is_trained = False
        self.training_metrics: dict[str, Any] = {}
        self.cv_scores: list[float] = []
        self.optimal_threshold: float = 0.5
        self._load()

    def _get_path(self, name: str) -> Path:
        return self.model_dir / name

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def extract_features(self, telemetry: dict) -> np.ndarray:
        f = {
            "gpu_util": telemetry.get("gpu_utilization", 0),
            "mem_util": telemetry.get("memory_utilization", 0),
            "temp": telemetry.get("temperature", 0),
            "power": telemetry.get("power_usage", 0),
            "clock": telemetry.get("clock_speed", 0),
            "ecc_err": telemetry.get("ecc_errors", 0),
            "retired_pages": telemetry.get("retired_pages", 0),
            "xid_err": telemetry.get("xid_errors", 0),
            "util_var": telemetry.get("utilization_variance", 0),
            "temp_var": telemetry.get("temperature_variance", 0),
            "avail_gpus": telemetry.get("available_gpus", 0),
            "total_gpus": telemetry.get("total_gpus", 1),
            "queue_len": telemetry.get("queue_length", 0),
            "job_fails": telemetry.get("job_failures", 0),
            "job_retries": telemetry.get("job_retries", 0),
            "avg_job_dur": telemetry.get("average_job_duration", 0),
        }
        f["gpu_util_ratio"] = f["gpu_util"] / 100.0
        f["mem_pressure"] = f["mem_util"] / 100.0
        f["temp_ratio"] = f["temp"] / 85.0
        f["error_rate"] = (f["ecc_err"] + f["xid_err"]) / 1000.0
        f["power_per_gpu"] = f["power"] / max(f["total_gpus"], 1)
        f["util_x_pressure"] = f["gpu_util_ratio"] * f["mem_pressure"]
        f["temp_x_power"] = f["temp_ratio"] * (f["power"] / 400.0)
        f["err_x_retry"] = f["error_rate"] * max(f["job_retries"], 1)
        f["mem_frag"] = 1.0 - (f["avail_gpus"] / max(f["total_gpus"], 1))
        f["stress_index"] = (f["gpu_util_ratio"] + f["mem_pressure"] + f["temp_ratio"]) / 3.0

        self.feature_names = list(f.keys())
        return np.array(list(f.values()))

    def generate_synthetic_data(self, n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        X = []
        y = []

        for _ in range(n_samples):
            t = {
                "gpu_utilization": rng.uniform(10, 98),
                "memory_utilization": rng.uniform(20, 98),
                "temperature": rng.uniform(35, 92),
                "power_usage": rng.uniform(50, 450),
                "clock_speed": rng.uniform(500, 2100),
                "ecc_errors": int(rng.poisson(2)),
                "retired_pages": int(rng.poisson(0.5)),
                "xid_errors": int(rng.poisson(1)),
                "utilization_variance": rng.uniform(0.02, 0.6),
                "temperature_variance": rng.uniform(0.05, 0.7),
                "available_gpus": int(rng.integers(0, 8)),
                "total_gpus": 8,
                "queue_length": int(rng.poisson(5)),
                "job_failures": int(rng.poisson(1)),
                "job_retries": int(rng.poisson(0.5)),
                "average_job_duration": rng.uniform(60, 7200),
            }
            feats = self.extract_features(t)

            risk = (
                feats[0] / 100 * 0.2
                + feats[1] / 100 * 0.2
                + feats[2] / 85 * 0.25
                + feats[5] / 20 * 0.1
                + feats[7] / 10 * 0.1
                + (1 - feats[10] / max(feats[11], 1)) * 0.1
                + feats[24] * 0.05
            )
            risk += rng.normal(0, 0.05)
            failure = int(risk > 0.45)

            X.append(feats)
            y.append(failure)

        return np.array(X), np.array(y)

    def train(
        self,
        telemetry_history: list[dict] | None = None,
        labels: list[int] | None = None,
        n_synthetic: int = 2000,
        use_cv: bool = True,
    ) -> dict[str, Any]:
        if telemetry_history and labels and len(telemetry_history) >= 50:
            X_real = np.array([self.extract_features(t) for t in telemetry_history])
            y_real = np.array(labels)
            X_synth, y_synth = self.generate_synthetic_data(max(n_synthetic, 2000 - len(telemetry_history)))
            X = np.vstack([X_real, X_synth])
            y = np.hstack([y_real, y_synth])
            logger.info("Training on %d real + %d synthetic = %d total", len(X_real), len(X_synth), len(X))
        else:
            X, y = self.generate_synthetic_data(n_synthetic)
            logger.info("Training on %d synthetic samples", len(X))

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Training RandomForest...")
        self.rf.fit(X_train, y_train)
        logger.info("Training GradientBoosting...")
        self.gb.fit(X_train, y_train)
        logger.info("Training NeuralNetwork...")
        self.nn.fit(X_train, y_train)
        logger.info("Training IsolationForest...")
        self.anomaly.fit(X_train)

        self.is_trained = True

        rf_pred = self.rf.predict(X_test)
        gb_pred = self.gb.predict(X_test)
        nn_pred = self.nn.predict(X_test)
        rf_prob = self.rf.predict_proba(X_test)[:, 1]
        gb_prob = self.gb.predict_proba(X_test)[:, 1]
        nn_prob = self.nn.predict_proba(X_test)[:, 1]

        ensemble_prob = (rf_prob + gb_prob + nn_prob) / 3.0

        precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_prob)
        f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_idx = int(np.argmax(f1_scores))
        self.optimal_threshold = float(thresholds[best_idx]) if len(thresholds) > best_idx else 0.5

        ensemble_pred = (ensemble_prob >= self.optimal_threshold).astype(int)

        self.training_metrics = {
            "accuracy": round(float(accuracy_score(y_test, ensemble_pred)), 4),
            "precision": round(float(precision_score(y_test, ensemble_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, ensemble_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, ensemble_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, ensemble_prob)), 4),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "test_samples": int(len(y_test)),
            "class_distribution": {"negative": int(np.sum(y == 0)), "positive": int(np.sum(y == 1))},
        }

        if use_cv:
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.rf, X_scaled, y, cv=cv, scoring="roc_auc")
                self.cv_scores = [round(float(s), 4) for s in cv_scores]
                self.training_metrics["cv_roc_auc_mean"] = round(float(np.mean(cv_scores)), 4)
                self.training_metrics["cv_roc_auc_std"] = round(float(np.std(cv_scores)), 4)
            except Exception as e:
                logger.warning("Cross-validation failed: %s", e)

        importance = {}
        if hasattr(self.rf, "feature_importances_"):
            rf_imp = dict(zip(self.feature_names, [float(v) for v in self.rf.feature_importances_]))
            importance["random_forest"] = dict(sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:10])
        if hasattr(self.gb, "feature_importances_"):
            gb_imp = dict(zip(self.feature_names, [float(v) for v in self.gb.feature_importances_]))
            importance["gradient_boosting"] = dict(sorted(gb_imp.items(), key=lambda x: x[1], reverse=True)[:10])

        cm = confusion_matrix(y_test, ensemble_pred).tolist()
        cr = classification_report(y_test, ensemble_pred, output_dict=True, zero_division=0)

        self._save()

        return {
            "status": "training_complete",
            "version": self.VERSION,
            "samples_total": len(X),
            "synthetic_samples": len(X_synth) if telemetry_history else len(X),
            "real_samples": len(telemetry_history) if telemetry_history else 0,
            "metrics": self.training_metrics,
            "cv_scores": self.cv_scores,
            "feature_importance": importance,
            "confusion_matrix": cm,
            "classification_report": cr,
        }

    def predict_proba_ensemble(self, features_scaled: np.ndarray) -> np.ndarray:
        rf_p = self.rf.predict_proba(features_scaled)[:, 1]
        gb_p = self.gb.predict_proba(features_scaled)[:, 1]
        nn_p = self.nn.predict_proba(features_scaled)[:, 1]
        return (rf_p + gb_p + nn_p) / 3.0

    def predict_failure(self, current_telemetry: dict) -> dict:
        feats = self.extract_features(current_telemetry)
        feats_scaled = self.scaler.transform([feats])

        rf_p = float(self.rf.predict_proba(feats_scaled)[0, 1]) if self.is_trained else 0.5
        gb_p = float(self.gb.predict_proba(feats_scaled)[0, 1]) if self.is_trained else 0.5
        nn_p = float(self.nn.predict_proba(feats_scaled)[0, 1]) if self.is_trained else 0.5
        ensemble_p = (rf_p + gb_p + nn_p) / 3.0

        is_anomaly = bool(self.anomaly.predict(feats_scaled)[0] == -1) if self.is_trained else False
        anomaly_score = float(self.anomaly.decision_function(feats_scaled)[0]) if self.is_trained else 0.0

        if is_anomaly:
            ensemble_p = min(1.0, ensemble_p * 1.25)

        prediction = bool(ensemble_p >= self.optimal_threshold)
        calibrated_p = self._calibrate_probability(ensemble_p, current_telemetry)

        risk_factors: list[str] = []
        temp = current_telemetry.get("temperature", 0)
        if temp > 80:
            risk_factors.append(f"Critical temperature: {temp}°C")
        elif temp > 70:
            risk_factors.append(f"Elevated temperature: {temp}°C")

        ecc = current_telemetry.get("ecc_errors", 0)
        if ecc > 15:
            risk_factors.append(f"Critical ECC errors: {ecc}")
        elif ecc > 5:
            risk_factors.append(f"Elevated ECC errors: {ecc}")

        xid = current_telemetry.get("xid_errors", 0)
        if xid > 5:
            risk_factors.append(f"XID errors detected: {xid}")

        mem = current_telemetry.get("memory_utilization", 0)
        if mem > 95:
            risk_factors.append(f"Critical memory pressure: {mem}%")
        elif mem > 85:
            risk_factors.append(f"High memory pressure: {mem}%")

        uv = current_telemetry.get("utilization_variance", 0)
        if uv > 0.4:
            risk_factors.append(f"Erratic utilization pattern: variance={uv:.2f}")

        power = current_telemetry.get("power_usage", 0)
        if power > 400:
            risk_factors.append(f"Power spike: {power}W")

        if is_anomaly:
            risk_factors.append("Anomalous behavior pattern detected by IsolationForest")

        model_agreement = sum([rf_p > 0.5, gb_p > 0.5, nn_p > 0.5])
        if model_agreement >= 2:
            confidence = max(rf_p, gb_p, nn_p) if ensemble_p > 0.5 else 1.0 - min(rf_p, gb_p, nn_p)
        else:
            confidence = 1.0 - abs(ensemble_p - 0.5) * 2

        if calibrated_p >= 0.7:
            recommendation = "Immediate action required: inspect node, migrate workloads, consider hardware RMA"
        elif calibrated_p >= 0.45:
            recommendation = "Prepare for potential failure: monitor closely, reduce load, schedule maintenance"
        else:
            recommendation = "System healthy: continue routine monitoring"

        return {
            "version": self.VERSION,
            "failure_predicted": prediction,
            "probability_raw": round(ensemble_p, 4),
            "probability_calibrated": round(calibrated_p, 4),
            "confidence": round(float(confidence), 4),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "model_probs": {
                "random_forest": round(rf_p, 4),
                "gradient_boosting": round(gb_p, 4),
                "neural_network": round(nn_p, 4),
            },
            "risk_factors": risk_factors,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "recommendation": recommendation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _calibrate_probability(self, raw_prob: float, telemetry: dict) -> float:
        penalty = 0.0
        if telemetry.get("temperature", 0) > 85:
            penalty += 0.08
        if telemetry.get("ecc_errors", 0) > 15:
            penalty += 0.06
        if telemetry.get("xid_errors", 0) > 8:
            penalty += 0.07
        if telemetry.get("memory_utilization", 0) > 95:
            penalty += 0.05
        if telemetry.get("retired_pages", 0) > 3:
            penalty += 0.04
        if telemetry.get("utilization_variance", 0) > 0.5:
            penalty += 0.03
        return min(1.0, raw_prob + penalty)

    def analyze_cluster(self, cluster_id: str, node_count: int = 8) -> dict:
        rng = random.Random(hash(cluster_id))
        results: dict[str, dict] = {}
        high_risk = 0
        medium_risk = 0
        total_risk = 0.0

        for i in range(node_count):
            base_risk = rng.uniform(0.1, 0.6)
            telemetry = {
                "gpu_utilization": rng.uniform(15, 98),
                "memory_utilization": rng.uniform(25, 98),
                "temperature": rng.uniform(35, 92),
                "power_usage": rng.uniform(50, 450),
                "clock_speed": rng.uniform(500, 2100),
                "ecc_errors": rng.randint(0, 25),
                "retired_pages": rng.randint(0, 5),
                "xid_errors": rng.randint(0, 12),
                "utilization_variance": rng.uniform(0.02, 0.65),
                "temperature_variance": rng.uniform(0.05, 0.7),
                "available_gpus": rng.randint(0, 8),
                "total_gpus": 8,
                "queue_length": rng.randint(0, 30),
                "job_failures": rng.randint(0, 8),
                "job_retries": rng.randint(0, 5),
                "average_job_duration": rng.uniform(60, 7200),
            }
            pred = self.predict_failure(telemetry)
            prob = pred["probability_calibrated"]
            total_risk += prob
            if prob > 0.7:
                high_risk += 1
            elif prob > 0.45:
                medium_risk += 1
            results[f"node-{i}"] = {
                "failure_probability": prob,
                "risk_factors": pred["risk_factors"],
                "recommendation": pred["recommendation"],
                "is_anomaly": pred["is_anomaly"],
                "model_probs": pred["model_probs"],
            }

        cluster_health = "critical" if high_risk > node_count * 0.3 else "degraded" if high_risk > 0 or medium_risk > node_count * 0.5 else "healthy"

        return {
            "status": "success",
            "version": self.VERSION,
            "cluster_id": cluster_id,
            "cluster_health": cluster_health,
            "nodes": results,
            "summary": {
                "total_nodes": node_count,
                "high_risk_nodes": high_risk,
                "medium_risk_nodes": medium_risk,
                "healthy_nodes": node_count - high_risk - medium_risk,
                "avg_risk_score": round(total_risk / max(node_count, 1), 4),
                "cluster_health": cluster_health,
            },
            "metrics": self.training_metrics if self.is_trained else {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_feature_importance(self) -> dict:
        if not self.is_trained or not self.feature_names:
            return {"status": "model_not_trained"}
        rf_imp = dict(zip(self.feature_names, self.rf.feature_importances_))
        gb_imp = dict(zip(self.feature_names, self.gb.feature_importances_))
        return {
            "random_forest_top10": dict(sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:10]),
            "gradient_boosting_top10": dict(sorted(gb_imp.items(), key=lambda x: x[1], reverse=True)[:10]),
            "ensemble_mean": dict(sorted(
                {k: (rf_imp[k] + gb_imp[k]) / 2 for k in rf_imp}.items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
        }

    def get_model_info(self) -> dict:
        return {
            "version": self.VERSION,
            "is_trained": self.is_trained,
            "feature_count": self.feature_count,
            "feature_names": self.feature_names,
            "models": ["random_forest", "gradient_boosting", "neural_network", "isolation_forest"],
            "training_metrics": self.training_metrics,
            "cv_scores": self.cv_scores,
            "optimal_threshold": round(self.optimal_threshold, 4),
            "model_dir": str(self.model_dir),
        }

    def _save(self) -> None:
        data = {
            "rf": self.rf,
            "gb": self.gb,
            "nn": self.nn,
            "anomaly": self.anomaly,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "training_metrics": self.training_metrics,
            "cv_scores": self.cv_scores,
            "optimal_threshold": self.optimal_threshold,
            "version": self.VERSION,
        }
        path = self._get_path("ensemble_failure_predictor.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        meta_path = self._get_path("ensemble_failure_predictor_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "version": self.VERSION,
                "is_trained": self.is_trained,
                "feature_count": self.feature_count,
                "training_metrics": self.training_metrics,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        logger.info("Ensemble model saved to %s", path)

    def _load(self) -> None:
        path = self._get_path("ensemble_failure_predictor.pkl")
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.rf = data["rf"]
                self.gb = data["gb"]
                self.nn = data["nn"]
                self.anomaly = data["anomaly"]
                self.scaler = data["scaler"]
                self.feature_names = data["feature_names"]
                self.is_trained = data["is_trained"]
                self.training_metrics = data.get("training_metrics", {})
                self.cv_scores = data.get("cv_scores", [])
                self.optimal_threshold = data.get("optimal_threshold", 0.5)
                logger.info("Ensemble model loaded from %s (v%s)", path, data.get("version", "unknown"))
            except Exception as e:
                logger.warning("Failed to load ensemble model: %s", e)
