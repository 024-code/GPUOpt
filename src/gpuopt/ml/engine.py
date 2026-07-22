from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .automl import AutoMLEngine
from .digital_twin_sim import DigitalTwinSimulationService
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class MLEngine:
    def __init__(self) -> None:
        from ..config import get_settings
        settings = get_settings()
        base = Path(settings.database_path).parent

        self.registry = ModelRegistry(base / "models" / "registry")
        self.automl = AutoMLEngine()
        self.twin_sim = DigitalTwinSimulationService()

        self._ensemble_predictor: Any = None

    @property
    def ensemble_predictor(self) -> Any:
        if self._ensemble_predictor is None:
            from ..predictor.ensemble_failure_predictor import EnsembleFailurePredictor
            self._ensemble_predictor = EnsembleFailurePredictor()
        return self._ensemble_predictor

    def predict_failure(self, telemetry: dict) -> dict:
        return self.ensemble_predictor.predict_failure(telemetry)

    def train_ensemble(self, telemetry_history: list[dict] | None = None,
                       labels: list[int] | None = None, n_synthetic: int = 2000) -> dict:
        result = self.ensemble_predictor.train(telemetry_history, labels, n_synthetic)

        self.registry.register_model(
            name="ensemble_failure_predictor",
            version=self.ensemble_predictor.VERSION,
            framework="scikit-learn",
            metrics=self.ensemble_predictor.training_metrics,
            params={
                "n_estimators_rf": 200,
                "n_estimators_gb": 150,
                "nn_layers": [64, 32, 16],
                "n_synthetic": n_synthetic,
            },
            description="Ensemble: RF + GB + NN with IsolationForest anomaly detection",
        )

        result["registry_entry"] = f"ensemble_failure_predictor v{self.ensemble_predictor.VERSION}"
        return result

    def analyze_cluster(self, cluster_id: str, node_count: int = 8) -> dict:
        return self.ensemble_predictor.analyze_cluster(cluster_id, node_count)

    def get_model_info(self) -> dict:
        return self.ensemble_predictor.get_model_info()

    def get_feature_importance(self) -> dict:
        return self.ensemble_predictor.get_feature_importance()

    def simulate(self, num_gpus: int = 8, gpu_model: str = "NVIDIA H100-SXM-80GB",
                 workload_type: str = "llm_inference", duration_steps: int = 60) -> dict:
        return self.twin_sim.simulate(num_gpus, gpu_model, workload_type, duration_steps)

    def simulate_failure(self, scenario: str = "thermal_runaway", num_gpus: int = 8) -> dict:
        return self.twin_sim.simulate_failure(scenario, num_gpus)

    def list_profiles(self) -> list[dict]:
        return self.twin_sim.list_profiles()

    def automl_random_search(self, model_type: str, n_iter: int = 20,
                              n_samples: int = 1000, cv_folds: int = 5) -> dict:
        X, y = self.ensemble_predictor.generate_synthetic_data(n_samples)
        return self.automl.random_search(model_type, X, y, n_iter, cv_folds)

    def automl_compare_models(self, n_samples: int = 1000) -> list[dict]:
        X, y = self.ensemble_predictor.generate_synthetic_data(n_samples)
        return self.automl.compare_all_models(X, y)

    def registry_list(self) -> dict:
        return self.registry.list_models()

    def registry_get(self, name: str, version: str | None = None) -> dict | None:
        return self.registry.get_model(name, version)

    def registry_promote(self, name: str, version: str, stage: str = "production") -> dict | None:
        if stage == "production":
            return self.registry.promote_to_production(name, version)
        return self.registry.promote_to_staging(name, version)

    def health(self) -> dict:
        return {
            "status": "healthy",
            "ensemble_trained": self.ensemble_predictor.is_trained,
            "ensemble_version": self.ensemble_predictor.VERSION,
            "registry": self.registry.health(),
            "automl": self.automl.health(),
            "twin_sim": self.twin_sim.health(),
        }
