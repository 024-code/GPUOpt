from __future__ import annotations

import itertools
import json
import logging
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample"],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
    "mlp": {
        "hidden_layer_sizes": [(32,), (64,), (64, 32), (64, 32, 16), (128, 64)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [200, 500, 1000],
        "alpha": [0.0001, 0.001, 0.01],
    },
}


class AutoMLEngine:
    def __init__(self, random_state: int = 42) -> None:
        self.rng = random.Random(random_state)
        self.np_rng = np.random.default_rng(random_state)
        self.results_dir = Path(__file__).resolve().parents[2] / "data" / "automl"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def random_search(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 20,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        maximize: bool = True,
    ) -> dict[str, Any]:
        if model_type not in SEARCH_SPACES:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(SEARCH_SPACES.keys())}")

        space = SEARCH_SPACES[model_type]
        results: list[dict] = []
        search_id = uuid.uuid4().hex[:8]

        logger.info("Starting random search for %s (%d iterations, %d-fold CV)", model_type, n_iter, cv_folds)

        for i in range(n_iter):
            params = {k: self.rng.choice(v) for k, v in space.items()}
            if model_type == "random_forest":
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(**params, random_state=42)
            elif model_type == "mlp":
                model = MLPClassifier(**params, random_state=42, early_stopping=True, validation_fraction=0.1)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            try:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))
            except Exception as e:
                logger.warning("Iteration %d failed: %s", i, e)
                continue

            model.fit(X, y)
            y_pred = model.predict(X)
            train_f1 = float(f1_score(y, y_pred, zero_division=0))
            train_precision = float(precision_score(y, y_pred, zero_division=0))
            train_recall = float(recall_score(y, y_pred, zero_division=0))

            results.append({
                "iteration": i + 1,
                "params": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in params.items()},
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
                "train_f1": round(train_f1, 4),
                "train_precision": round(train_precision, 4),
                "train_recall": round(train_recall, 4),
            })

        results.sort(key=lambda r: r["mean_score"], reverse=maximize)
        best = results[0] if results else {}

        trial_data = {
            "search_id": search_id,
            "model_type": model_type,
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "n_samples": len(X),
            "results": results,
            "best_params": best.get("params", {}),
            "best_score": best.get("mean_score"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        path = self.results_dir / f"hpo_{model_type}_{search_id}.json"
        path.write_text(json.dumps(trial_data, indent=2, default=str))
        logger.info("HPO results saved to %s (best score: %s)", path, best.get("mean_score"))

        return trial_data

    def grid_search(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> dict[str, Any]:
        if model_type not in SEARCH_SPACES:
            raise ValueError(f"Unknown model type: {model_type}")
        space = param_grid or SEARCH_SPACES[model_type]

        keys = list(space.keys())
        values = list(space.values())
        total = len(list(itertools.product(*values)))
        logger.info("Grid search for %s with %d combinations", model_type, total)

        results: list[dict] = []
        search_id = uuid.uuid4().hex[:8]

        for i, combo in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combo))
            if model_type == "random_forest":
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(**params, random_state=42)
            elif model_type == "mlp":
                model = MLPClassifier(**params, random_state=42, early_stopping=True, validation_fraction=0.1)
            else:
                continue

            try:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))
            except Exception as e:
                logger.warning("Combination %d failed: %s", i, e)
                continue

            results.append({
                "iteration": i + 1,
                "params": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in params.items()},
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
            })

            if (i + 1) % 10 == 0:
                logger.info("Grid search %s: %d/%d complete", model_type, i + 1, total)

        results.sort(key=lambda r: r["mean_score"], reverse=True)
        best = results[0] if results else {}

        trial_data = {
            "search_id": search_id,
            "model_type": model_type,
            "type": "grid_search",
            "total_combinations": total,
            "completed": len(results),
            "cv_folds": cv_folds,
            "scoring": scoring,
            "results": results,
            "best_params": best.get("params", {}),
            "best_score": best.get("mean_score"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        path = self.results_dir / f"grid_{model_type}_{search_id}.json"
        path.write_text(json.dumps(trial_data, indent=2, default=str))
        return trial_data

    def bayesian_optimization(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 30,
        n_initial: int = 5,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> dict[str, Any]:
        if model_type not in SEARCH_SPACES:
            raise ValueError(f"Unknown model type: {model_type}")

        space = SEARCH_SPACES[model_type]
        search_id = uuid.uuid4().hex[:8]
        all_results: list[dict] = []

        def sample_params() -> dict:
            return {k: self.rng.choice(v) for k, v in space.items()}

        def evaluate_params(params: dict) -> float:
            if model_type == "random_forest":
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(**params, random_state=42)
            elif model_type == "mlp":
                model = MLPClassifier(**params, random_state=42, early_stopping=True, validation_fraction=0.1)
            else:
                return 0.0
            try:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                return float(np.mean(scores))
            except Exception:
                return 0.0

        for i in range(n_initial):
            params = sample_params()
            score = evaluate_params(params)
            all_results.append({"iteration": i + 1, "params": params, "score": score, "phase": "exploration"})

        for i in range(n_iter - n_initial):
            scores_arr = np.array([r["score"] for r in all_results])
            best_score = float(np.max(scores_arr))
            best_idx = int(np.argmax(scores_arr))
            best_params = all_results[best_idx]["params"]

            if self.rng.random() < 0.7:
                mutated = {}
                for k, v_list in space.items():
                    if self.rng.random() < 0.4:
                        mutated[k] = self.rng.choice(v_list)
                    else:
                        mutated[k] = best_params[k]
                params = mutated
            else:
                params = sample_params()

            score = evaluate_params(params)
            phase = "refinement" if score > best_score else "exploration"
            all_results.append({
                "iteration": n_initial + i + 1,
                "params": params,
                "score": score,
                "phase": phase,
            })

        all_results.sort(key=lambda r: r["score"], reverse=True)
        best = all_results[0]

        trial_data = {
            "search_id": search_id,
            "model_type": model_type,
            "type": "bayesian_optimization",
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "results": all_results,
            "best_params": best["params"],
            "best_score": round(best["score"], 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        path = self.results_dir / f"bayes_{model_type}_{search_id}.json"
        path.write_text(json.dumps(trial_data, indent=2, default=str))
        return trial_data

    def compare_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> list[dict[str, Any]]:
        configs = [
            ("random_forest", {"n_estimators": 200, "max_depth": 10, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}),
            ("gradient_boosting", {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}),
            ("mlp", {"hidden_layer_sizes": (64, 32), "activation": "relu", "max_iter": 500, "random_state": 42, "early_stopping": True}),
        ]

        results = []
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, params in configs:
            if name == "random_forest":
                model = RandomForestClassifier(**params)
            elif name == "gradient_boosting":
                model = GradientBoostingClassifier(**params)
            elif name == "mlp":
                model = MLPClassifier(**params)
            else:
                continue

            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
                model.fit(X, y)
                y_pred = model.predict(X)
                results.append({
                    "model": name,
                    "roc_auc_mean": round(float(np.mean(scores)), 4),
                    "roc_auc_std": round(float(np.std(scores)), 4),
                    "train_f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
                    "train_precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
                    "train_recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
                })
            except Exception as e:
                logger.warning("Model %s failed: %s", name, e)

        results.sort(key=lambda r: r["roc_auc_mean"], reverse=True)
        return results

    def health(self) -> dict:
        return {
            "status": "healthy",
            "search_spaces": list(SEARCH_SPACES.keys()),
            "results_dir": str(self.results_dir),
            "methods": ["random_search", "grid_search", "bayesian_optimization", "compare_all_models"],
        }
