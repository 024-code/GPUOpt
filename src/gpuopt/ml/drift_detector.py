from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from gpuopt.ml.features import extract_state_features, extract_telemetry_features
from gpuopt.schemas import (
    ClusterStateData,
    DriftItem,
    DriftSeverity,
    GPUDeviceState,
    NodeState,
)

logger = logging.getLogger(__name__)

_DRIFT_HISTORY_WINDOW = 500


class DriftDetector:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path("./data/ml")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._baseline: dict[str, float] = {}
        self._baseline_set_at: datetime | None = None
        self._history: list[dict[str, float]] = []
        self._ewma: dict[str, float] = {}
        self._ewma_lambda: float = 0.3
        self._control_limits: dict[str, tuple[float, float, float]] = {}
        self._outlier_scaler: dict[str, float] = {}
        self._load()

    def _state_path(self) -> Path:
        return self.model_dir / "drift_state.json"

    def _model_path(self) -> Path:
        return self.model_dir / "drift_model.pkl"

    def _load(self) -> None:
        state_path = self._state_path()
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                self._baseline = data.get("baseline", {})
                self._ewma = data.get("ewma", {})
                self._control_limits = {
                    k: tuple(v) for k, v in data.get("control_limits", {}).items()
                }
                self._history = data.get("history", [])
                if data.get("baseline_set_at"):
                    self._baseline_set_at = datetime.fromisoformat(data["baseline_set_at"])
            except Exception as exc:
                logger.warning("Failed to load drift state: %s", exc)

    def _save(self) -> None:
        try:
            data = {
                "baseline": self._baseline,
                "ewma": self._ewma,
                "control_limits": {k: list(v) for k, v in self._control_limits.items()},
                "history": self._history[-_DRIFT_HISTORY_WINDOW:],
                "baseline_set_at": self._baseline_set_at.isoformat() if self._baseline_set_at else None,
            }
            self._state_path().write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save drift state: %s", exc)

    def set_baseline(self, state: ClusterStateData) -> None:
        feats = extract_state_features(state)
        telemetry = getattr(state, "telemetry", None)
        if telemetry:
            try:
                feats.update(extract_telemetry_features(telemetry))
            except Exception:
                pass
        self._baseline = feats
        self._ewma = {k: v for k, v in feats.items()}
        self._baseline_set_at = state.collected_at
        self._control_limits = {}
        self._history = [feats]
        self._save()
        logger.info("Drift baseline set with %d features", len(feats))

    def update(self, state: ClusterStateData) -> None:
        feats = extract_state_features(state)
        telemetry = getattr(state, "telemetry", None)
        if telemetry:
            try:
                feats.update(extract_telemetry_features(telemetry))
            except Exception:
                pass
        self._history.append(feats)
        if len(self._history) > _DRIFT_HISTORY_WINDOW:
            self._history = self._history[-_DRIFT_HISTORY_WINDOW:]
        if not self._ewma:
            self._ewma = {k: v for k, v in feats.items()}
        else:
            for k, v in feats.items():
                if k in self._ewma:
                    self._ewma[k] = self._ewma_lambda * v + (1 - self._ewma_lambda) * self._ewma[k]
                else:
                    self._ewma[k] = v
        self._update_control_limits()
        self._save()

    def _update_control_limits(self) -> None:
        if len(self._history) < 5:
            return
        keys = set()
        for h in self._history:
            keys.update(h.keys())
        for key in keys:
            values = [h.get(key, 0.0) for h in self._history[-50:] if key in h]
            if len(values) < 5:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std > 0:
                cl = mean
                ucl = mean + 3 * std
                lcl = max(mean - 3 * std, 0)
                self._control_limits[key] = (cl, lcl, ucl)

    def detect_node_drift(
        self,
        twin_state: ClusterStateData,
        actual_state: ClusterStateData,
    ) -> list[DriftItem]:
        drifts: list[DriftItem] = []
        twin_nodes = {n.name: n for n in twin_state.nodes}
        actual_nodes = {n.name: n for n in actual_state.nodes}
        twin_names = set(twin_nodes.keys())
        actual_names = set(actual_nodes.keys())

        for name in sorted(twin_names - actual_names):
            drifts.append(DriftItem(
                resource=f"node/{name}",
                property="status",
                twin_value="present",
                actual_value="absent",
                severity=DriftSeverity.HIGH,
                message=f"Node {name} is in the twin but missing from the actual cluster",
            ))
        for name in sorted(actual_names - twin_names):
            drifts.append(DriftItem(
                resource=f"node/{name}",
                property="status",
                twin_value="absent",
                actual_value="present",
                severity=DriftSeverity.MEDIUM,
                message=f"Node {name} exists in the actual cluster but not in the twin",
            ))
        for name in sorted(twin_names & actual_names):
            tn = twin_nodes[name]
            an = actual_nodes[name]
            if tn.status != an.status:
                drifts.append(DriftItem(
                    resource=f"node/{name}",
                    property="status",
                    twin_value=tn.status,
                    actual_value=an.status,
                    severity=DriftSeverity.MEDIUM,
                    message=f"Node {name} status changed: twin={tn.status} actual={an.status}",
                ))
            drifts.extend(self._detect_gpu_drift(name, tn, an))
        return drifts

    def _detect_gpu_drift(self, node_name: str, twin_node: NodeState, actual_node: NodeState) -> list[DriftItem]:
        drifts: list[DriftItem] = []
        twin_gpus = {g.index: g for g in twin_node.gpu_devices}
        actual_gpus = {g.index: g for g in actual_node.gpu_devices}
        for idx in sorted(set(twin_gpus.keys()) | set(actual_gpus.keys())):
            tg = twin_gpus.get(idx)
            ag = actual_gpus.get(idx)
            if tg is None and ag is not None:
                drifts.append(DriftItem(
                    resource=f"node/{node_name}/gpu-{idx}",
                    property="status",
                    twin_value="absent",
                    actual_value="present",
                    severity=DriftSeverity.MEDIUM,
                    message=f"GPU {idx} appeared on {node_name} since twin was synced",
                ))
            elif tg is not None and ag is None:
                drifts.append(DriftItem(
                    resource=f"node/{node_name}/gpu-{idx}",
                    property="status",
                    twin_value="present",
                    actual_value="absent",
                    severity=DriftSeverity.HIGH,
                    message=f"GPU {idx} on {node_name} is missing from actual cluster",
                ))
            elif tg is not None and ag is not None:
                self._check_gpu_metric_drift(drifts, node_name, idx, tg, ag, "memory_used_bytes")
                self._check_gpu_metric_drift(drifts, node_name, idx, tg, ag, "memory_total_bytes")
        return drifts

    def _check_gpu_metric_drift(
        self,
        drifts: list[DriftItem],
        node_name: str,
        gpu_idx: int,
        twin_gpu: GPUDeviceState,
        actual_gpu: GPUDeviceState,
        metric: str,
    ) -> None:
        twin_val = getattr(twin_gpu, metric, 0)
        actual_val = getattr(actual_gpu, metric, 0)
        if twin_val == actual_val:
            return
        diff = abs(twin_val - actual_val)
        threshold = 0.1 * max(twin_val, actual_val)
        z_score = self._compute_z_score_for_metric(metric, twin_val, actual_val)
        sev = self._severity_from_drift(metric, diff, twin_val, z_score)
        if diff > threshold or abs(z_score) > 2.0:
            drifts.append(DriftItem(
                resource=f"node/{node_name}/gpu-{gpu_idx}",
                property=metric,
                twin_value=str(twin_val),
                actual_value=str(actual_val),
                severity=sev,
                message=(
                    f"GPU {gpu_idx} {metric} drifted by {diff / (1024**2):.0f} MiB "
                    f"(z={z_score:.2f}, severity={sev.value})"
                ),
            ))

    def _compute_z_score_for_metric(self, metric: str, twin_val: float, actual_val: float) -> float:
        key = f"gpu_{metric}"
        if key in self._control_limits:
            _, _, ucl = self._control_limits[key]
            std = (ucl - twin_val) / 3.0 if ucl > twin_val else max(abs(twin_val) * 0.1, 1.0)
            return (actual_val - twin_val) / max(std, 1.0)
        hist_values = [
            h.get(key, 0.0) for h in self._history[-20:]
            if key in h and h.get(key, 0.0) > 0
        ]
        if len(hist_values) >= 3:
            mean = float(np.mean(hist_values))
            std = float(np.std(hist_values))
            if std > 0:
                return (actual_val - mean) / std
        return 0.0

    def _severity_from_drift(
        self, metric: str, diff: float, baseline: float,
        z_score: float,
    ) -> DriftSeverity:
        abs_z = abs(z_score)
        if abs_z > 4.0:
            return DriftSeverity.CRITICAL
        if abs_z > 3.0:
            return DriftSeverity.HIGH
        if abs_z > 2.0:
            return DriftSeverity.MEDIUM
        if abs_z > 1.0:
            return DriftSeverity.LOW
        pct = diff / max(baseline, 1) * 100
        if pct > 50:
            return DriftSeverity.HIGH
        if pct > 25:
            return DriftSeverity.MEDIUM
        if pct > 10:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def detect_feature_anomaly(self, state: ClusterStateData) -> list[DriftItem]:
        drifts: list[DriftItem] = []
        if not self._baseline:
            return drifts
        feats = extract_state_features(state)
        telemetry = getattr(state, "telemetry", None)
        if telemetry:
            try:
                feats.update(extract_telemetry_features(telemetry))
            except Exception:
                pass
        for key, actual_val in feats.items():
            baseline_val = self._baseline.get(key)
            if baseline_val is None:
                continue
            diff = abs(actual_val - baseline_val)
            if baseline_val == 0 and actual_val == 0:
                continue
            pct_change = diff / max(abs(baseline_val), 1e-10) * 100
            z_score = 0.0
            if key in self._control_limits:
                _, lcl, ucl = self._control_limits[key]
                std = max((ucl - lcl) / 6.0, 1e-10)
                z_score = (actual_val - baseline_val) / std
            elif len(self._history) >= 5:
                hist = [h.get(key, baseline_val) for h in self._history[-20:] if key in h]
                if len(hist) >= 3:
                    z_score = (actual_val - float(np.mean(hist))) / max(float(np.std(hist)), 1e-10)
            if abs(z_score) > 2.0 or pct_change > 30:
                sev = self._severity_from_drift(key, diff, baseline_val, z_score)
                drifts.append(DriftItem(
                    resource=f"cluster/{key}",
                    property=key,
                    twin_value=f"{baseline_val:.2f}",
                    actual_value=f"{actual_val:.2f}",
                    severity=sev,
                    message=(
                        f"Feature {key} drifted {pct_change:.1f}% "
                        f"(baseline={baseline_val:.2f}, actual={actual_val:.2f}, z={z_score:.2f})"
                    ),
                ))
        return drifts

    def get_control_limits(self) -> dict[str, dict[str, float]]:
        return {
            k: {"center": cl, "upper": ucl, "lower": lcl}
            for k, (cl, lcl, ucl) in self._control_limits.items()
        }

    def get_baseline(self) -> dict[str, float]:
        return dict(self._baseline)

    def reset(self) -> None:
        self._baseline.clear()
        self._baseline_set_at = None
        self._history.clear()
        self._ewma.clear()
        self._control_limits.clear()
        for p in [self._state_path()]:
            if p.exists():
                p.unlink()
        logger.info("Drift detector reset")
