from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, registry_dir: str | Path | None = None) -> None:
        if registry_dir is None:
            from ..config import get_settings
            settings = get_settings()
            registry_dir = Path(settings.database_path).parent / "models" / "registry"
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    def _index_path(self) -> Path:
        return self.registry_dir / "index.json"

    def _load_index(self) -> dict:
        path = self._index_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                return {"models": {}, "experiments": []}
        return {"models": {}, "experiments": []}

    def _save_index(self) -> None:
        self._index_path().write_text(json.dumps(self._index, indent=2, default=str))

    def register_model(
        self,
        name: str,
        version: str,
        framework: str = "scikit-learn",
        metrics: dict | None = None,
        params: dict | None = None,
        model_path: str | None = None,
        description: str = "",
    ) -> dict:
        model_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        entry = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "framework": framework,
            "description": description,
            "metrics": metrics or {},
            "params": params or {},
            "model_path": model_path,
            "created_at": timestamp,
            "status": "registered",
        }

        if name not in self._index["models"]:
            self._index["models"][name] = {"versions": []}
        self._index["models"][name]["versions"].append(entry)
        self._index["models"][name]["latest"] = version
        self._save_index()

        logger.info("Registered model %s v%s (id=%s)", name, version, model_id)
        return entry

    def promote_to_staging(self, name: str, version: str) -> dict | None:
        model = self._find_version(name, version)
        if model:
            model["status"] = "staging"
            self._save_index()
            return model
        return None

    def promote_to_production(self, name: str, version: str) -> dict | None:
        old_prod = self._get_production_version(name)
        model = self._find_version(name, version)
        if model:
            for v in self._index["models"].get(name, {}).get("versions", []):
                if v["status"] == "production":
                    v["status"] = "archived"
            model["status"] = "production"
            model["promoted_at"] = datetime.now(timezone.utc).isoformat()
            self._save_index()
            result = {
                "previous_production": old_prod,
                "new_production": model,
                "message": f"Model {name} v{version} promoted to production",
            }
            return result
        return None

    def _find_version(self, name: str, version: str) -> dict | None:
        versions = self._index["models"].get(name, {}).get("versions", [])
        for v in versions:
            if v["version"] == version:
                return v
        return None

    def _get_production_version(self, name: str) -> dict | None:
        versions = self._index["models"].get(name, {}).get("versions", [])
        for v in versions:
            if v["status"] == "production":
                return v
        return None

    def get_model(self, name: str, version: str | None = None) -> dict | None:
        if version:
            return self._find_version(name, version)
        latest = self._index["models"].get(name, {}).get("latest")
        if latest:
            return self._find_version(name, latest)
        return None

    def list_models(self) -> dict[str, Any]:
        return {
            name: {
                "versions": len(info["versions"]),
                "latest": info.get("latest"),
                "production": self._get_production_version(name),
            }
            for name, info in self._index["models"].items()
        }

    def get_model_history(self, name: str) -> list[dict]:
        return self._index["models"].get(name, {}).get("versions", [])

    def log_experiment(
        self,
        name: str,
        params: dict | None = None,
        metrics: dict | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        exp = {
            "experiment_id": str(uuid.uuid4()),
            "name": name,
            "params": params or {},
            "metrics": metrics or {},
            "tags": tags or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._index["experiments"].append(exp)
        self._save_index()
        return exp

    def list_experiments(self, name: str | None = None, limit: int = 50) -> list[dict]:
        exps = self._index["experiments"]
        if name:
            exps = [e for e in exps if e["name"] == name]
        return sorted(exps, key=lambda e: e["timestamp"], reverse=True)[:limit]

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict]:
        exp_map = {e["experiment_id"]: e for e in self._index["experiments"]}
        return [exp_map[eid] for eid in experiment_ids if eid in exp_map]

    def store_artifact(self, name: str, version: str, artifact_path: Path, artifact_type: str = "model") -> dict:
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / artifact_path.name
        shutil.copy2(str(artifact_path), str(dest))
        entry = {
            "artifact_type": artifact_type,
            "source": str(artifact_path),
            "destination": str(dest),
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        model = self._find_version(name, version)
        if model:
            if "artifacts" not in model:
                model["artifacts"] = []
            model["artifacts"].append(entry)
            self._save_index()
        return entry

    def get_metrics_history(self, name: str, metric_key: str | None = None) -> list[dict]:
        versions = self._index["models"].get(name, {}).get("versions", [])
        history = []
        for v in versions:
            if metric_key:
                val = v.get("metrics", {}).get(metric_key)
                if val is not None:
                    history.append({"version": v["version"], metric_key: val, "created_at": v["created_at"], "status": v["status"]})
            else:
                history.append({"version": v["version"], "metrics": v.get("metrics", {}), "created_at": v["created_at"], "status": v["status"]})
        return sorted(history, key=lambda h: h["created_at"])

    def health(self) -> dict:
        model_count = sum(len(info["versions"]) for info in self._index["models"].values())
        return {
            "status": "healthy",
            "model_count": model_count,
            "experiment_count": len(self._index["experiments"]),
            "registry_dir": str(self.registry_dir),
        }
