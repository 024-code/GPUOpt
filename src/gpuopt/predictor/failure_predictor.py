from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FailurePredictor:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_trained = False
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            from ..config import get_settings
            settings = get_settings()
            base = settings.database_path.parent
            self.model_path = base / "models" / "failure_predictor.pkl"
        self.load_model()

    def extract_features(self, telemetry: dict) -> np.ndarray:
        features = {
            "gpu_utilization": telemetry.get("gpu_utilization", 0),
            "memory_utilization": telemetry.get("memory_utilization", 0),
            "temperature": telemetry.get("temperature", 0),
            "power_usage": telemetry.get("power_usage", 0),
            "clock_speed": telemetry.get("clock_speed", 0),
            "ecc_errors": telemetry.get("ecc_errors", 0),
            "retired_pages": telemetry.get("retired_pages", 0),
            "xid_errors": telemetry.get("xid_errors", 0),
            "utilization_variance": telemetry.get("utilization_variance", 0),
            "temperature_variance": telemetry.get("temperature_variance", 0),
            "available_gpus": telemetry.get("available_gpus", 0),
            "total_gpus": telemetry.get("total_gpus", 1),
            "queue_length": telemetry.get("queue_length", 0),
            "job_failures": telemetry.get("job_failures", 0),
            "job_retries": telemetry.get("job_retries", 0),
            "average_job_duration": telemetry.get("average_job_duration", 0),
        }

        features["gpu_utilization_ratio"] = features["gpu_utilization"] / 100.0
        features["memory_pressure"] = features["memory_utilization"] / 100.0
        features["temperature_ratio"] = features["temperature"] / 85.0
        features["error_rate"] = (features["ecc_errors"] + features["xid_errors"]) / 1000.0

        self.feature_names = list(features.keys())
        return np.array(list(features.values()))

    def train(self, telemetry_history: list[dict], labels: list[int]) -> dict:
        if len(telemetry_history) < 100:
            msg = f"Need at least 100 samples for training (got {len(telemetry_history)})"
            logger.warning(msg)
            return {"status": "insufficient_data", "message": msg}

        X = np.array([self.extract_features(t) for t in telemetry_history])
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Training Random Forest on %d samples...", len(X_train))
        self.model.fit(X_train, y_train)

        logger.info("Training Isolation Forest for anomaly detection...")
        self.anomaly_detector.fit(X_train)

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred).tolist()

        self.is_trained = True

        importance: dict[str, float] = {}
        if hasattr(self.model, "feature_importances_"):
            importance = dict(
                zip(self.feature_names, [float(v) for v in self.model.feature_importances_])
            )

        self.save_model()

        return {
            "status": "training_complete",
            "samples": len(X_train),
            "accuracy": report.get("accuracy", 0),
            "classification_report": report,
            "confusion_matrix": matrix,
            "feature_importance": importance,
        }

    def predict_failure(self, current_telemetry: dict) -> dict:
        if not self.is_trained:
            return {
                "failure_predicted": False,
                "probability": 0.5,
                "risk_factors": ["Model not trained yet"],
                "recommendation": "Train model with historical data first",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        features = self.extract_features(current_telemetry)
        features_scaled = self.scaler.transform([features])

        rf_prediction = self.model.predict(features_scaled)[0]
        rf_probability = self.model.predict_proba(features_scaled)[0]
        anomaly_score = float(self.anomaly_detector.decision_function(features_scaled)[0])
        is_anomaly = bool(self.anomaly_detector.predict(features_scaled)[0] == -1)

        probability = float(rf_probability[1]) if len(rf_probability) > 1 else 0.5
        if is_anomaly:
            probability = min(1.0, probability * 1.3)

        risk_factors: list[str] = []

        temp = current_telemetry.get("temperature", 0)
        if temp > 80:
            risk_factors.append(f"High temperature: {temp}°C")

        ecc = current_telemetry.get("ecc_errors", 0)
        if ecc > 10:
            risk_factors.append(f"High ECC errors: {ecc}")

        xid = current_telemetry.get("xid_errors", 0)
        if xid > 5:
            risk_factors.append(f"XID errors detected: {xid}")

        mem = current_telemetry.get("memory_utilization", 0)
        if mem > 90:
            risk_factors.append(f"High memory pressure: {mem}%")

        uv = current_telemetry.get("utilization_variance", 0)
        if uv > 0.3:
            risk_factors.append(f"Erratic utilization: variance={uv:.2f}")

        if is_anomaly:
            risk_factors.append("Anomalous behavior detected")

        if probability > 0.7:
            recommendation = "Immediate action recommended: check node health, consider migrating workloads"
        elif probability > 0.4:
            recommendation = "Monitor closely: prepare for potential failure"
        else:
            recommendation = "System appears healthy, continue monitoring"

        return {
            "failure_predicted": bool(rf_prediction == 1) or is_anomaly,
            "probability": probability,
            "confidence": float(rf_probability[1] if len(rf_probability) > 1 else 0.5),
            "risk_factors": risk_factors,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "recommendation": recommendation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def analyze_cluster(self, cluster_id: str, node_count: int = 4) -> dict:
        import random as _random

        results: dict[str, dict] = {}
        for i in range(node_count):
            telemetry = {
                "gpu_utilization": _random.uniform(20, 95),
                "memory_utilization": _random.uniform(30, 98),
                "temperature": _random.uniform(40, 90),
                "power_usage": _random.uniform(100, 400),
                "clock_speed": _random.uniform(1000, 2000),
                "ecc_errors": _random.randint(0, 20),
                "retired_pages": _random.randint(0, 5),
                "xid_errors": _random.randint(0, 10),
                "utilization_variance": _random.uniform(0.05, 0.5),
                "temperature_variance": _random.uniform(0.1, 0.6),
                "available_gpus": _random.randint(0, 8),
                "total_gpus": 8,
                "queue_length": _random.randint(0, 20),
                "job_failures": _random.randint(0, 5),
                "job_retries": _random.randint(0, 3),
                "average_job_duration": _random.uniform(60, 3600),
            }
            result = self.predict_failure(telemetry)
            results[f"node-{i}"] = {
                "failure_probability": result["probability"],
                "risk_factors": result["risk_factors"],
                "recommendation": result["recommendation"],
                "is_anomaly": result.get("is_anomaly", False),
            }

        high_risk = [n for n, r in results.items() if r["failure_probability"] > 0.7]
        medium_risk = [
            n for n, r in results.items() if 0.4 < r["failure_probability"] <= 0.7
        ]

        return {
            "status": "success",
            "cluster_id": cluster_id,
            "nodes": results,
            "summary": {
                "total_nodes": len(results),
                "high_risk_nodes": len(high_risk),
                "medium_risk_nodes": len(medium_risk),
                "high_risk_node_ids": high_risk,
                "medium_risk_node_ids": medium_risk,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "anomaly_detector": self.anomaly_detector,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "is_trained": self.is_trained,
                },
                f,
            )
        logger.info("Failure predictor model saved to %s", self.model_path)

    def load_model(self) -> None:
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.anomaly_detector = data["anomaly_detector"]
                self.scaler = data["scaler"]
                self.feature_names = data["feature_names"]
                self.is_trained = data["is_trained"]
            logger.info("Failure predictor model loaded from %s", self.model_path)
        else:
            logger.info("No existing failure predictor model found, starting fresh")
