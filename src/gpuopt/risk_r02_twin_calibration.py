from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .schemas import (
    TwinConfidenceLimits,
    TwinFallbackMode,
    WorkloadFamilyProfile,
)

logger = logging.getLogger(__name__)

# Predefined workload family profiles
WORKLOAD_FAMILIES: dict[str, dict[str, Any]] = {
    "llm_training": {
        "description": "Large language model training (GPT, Llama, etc.)",
        "typical_gpu_count": 8, "typical_memory_gb": 80.0, "typical_duration_minutes": 1440.0,
        "parallelism_efficiency": 0.75, "checkpoint_compatible": True, "preemptible": True,
    },
    "llm_inference": {
        "description": "LLM inference serving",
        "typical_gpu_count": 1, "typical_memory_gb": 40.0, "typical_duration_minutes": 999999.0,
        "parallelism_efficiency": 1.0, "checkpoint_compatible": False, "preemptible": False,
    },
    "cnn_training": {
        "description": "CNN/vision model training",
        "typical_gpu_count": 4, "typical_memory_gb": 24.0, "typical_duration_minutes": 480.0,
        "parallelism_efficiency": 0.85, "checkpoint_compatible": True, "preemptible": True,
    },
    "batch_inference": {
        "description": "Batch inference jobs",
        "typical_gpu_count": 1, "typical_memory_gb": 16.0, "typical_duration_minutes": 60.0,
        "parallelism_efficiency": 1.0, "checkpoint_compatible": False, "preemptible": True,
    },
    "data_processing": {
        "description": "GPU-accelerated data processing (RAPIDS, etc.)",
        "typical_gpu_count": 2, "typical_memory_gb": 32.0, "typical_duration_minutes": 120.0,
        "parallelism_efficiency": 0.9, "checkpoint_compatible": False, "preemptible": True,
    },
    "hpo": {
        "description": "Hyperparameter optimization trials",
        "typical_gpu_count": 1, "typical_memory_gb": 16.0, "typical_duration_minutes": 30.0,
        "parallelism_efficiency": 0.95, "checkpoint_compatible": False, "preemptible": True,
    },
}


class WorkloadFamilyCalibrator:
    def __init__(self) -> None:
        self._profiles: dict[str, WorkloadFamilyProfile] = {}

    def get_family(self, workload: dict) -> str:
        framework = (workload.get("framework") or "").lower()
        job_type = (workload.get("type") or "").lower()
        duration = workload.get("max_duration_minutes", 120)
        gpus = workload.get("gpu_required", 1)

        if "llm" in job_type or "gpt" in job_type or "llama" in job_type:
            if duration > 120:
                return "llm_training"
            return "llm_inference"
        if "cnn" in job_type or "vision" in job_type or "image" in job_type:
            return "cnn_training"
        if "batch" in job_type or "inference" in job_type or "serve" in job_type:
            return "batch_inference"
        if "data" in job_type or "etl" in job_type or "rapids" in job_type:
            return "data_processing"
        if "hpo" in job_type or "tune" in job_type or "search" in job_type:
            return "hpo"
        if framework in ("pytorch", "tensorflow", "jax") and gpus > 1 and duration > 60:
            return "llm_training" if gpus >= 4 else "cnn_training"
        return "llm_inference" if "serve" in framework else "batch_inference"

    def get_profile(self, family: str) -> WorkloadFamilyProfile:
        if family not in self._profiles:
            defaults = WORKLOAD_FAMILIES.get(family, WORKLOAD_FAMILIES["batch_inference"])
            self._profiles[family] = WorkloadFamilyProfile(
                family=family,
                description=defaults["description"],
                typical_gpu_count=defaults["typical_gpu_count"],
                typical_memory_gb=defaults["typical_memory_gb"],
                typical_duration_minutes=defaults["typical_duration_minutes"],
                parallelism_efficiency=defaults["parallelism_efficiency"],
                checkpoint_compatible=defaults["checkpoint_compatible"],
                preemptible=defaults["preemptible"],
                calibration_bias=random.gauss(0, 0.05),
                calibration_variance=random.uniform(0.05, 0.15),
                sample_count=random.randint(10, 500),
            )
        return self._profiles[family]

    def record_outcome(self, family: str, simulated: float, actual: float) -> None:
        profile = self.get_profile(family)
        bias = simulated - actual
        n = profile.sample_count
        profile.calibration_bias = (profile.calibration_bias * n + bias) / (n + 1)
        profile.calibration_variance = (
            profile.calibration_variance * n + abs(bias - profile.calibration_bias)
        ) / (n + 1)
        profile.sample_count = n + 1

    def calibrate_prediction(self, family: str, raw_prediction: float) -> tuple[float, float]:
        profile = self.get_profile(family)
        calibrated = raw_prediction - profile.calibration_bias
        noise = random.gauss(0, profile.calibration_variance)
        return calibrated + noise, profile.calibration_variance

    def list_families(self) -> list[str]:
        return list(WORKLOAD_FAMILIES.keys())

    def get_all_profiles(self) -> list[WorkloadFamilyProfile]:
        return [self.get_profile(f) for f in WORKLOAD_FAMILIES]


class TwinConfidenceCalculator:
    def calculate(self, twin_id: str, profile: WorkloadFamilyProfile,
                 prediction_interval: float = 95.0) -> TwinConfidenceLimits:
        variance = profile.calibration_variance
        z = {80: 1.28, 90: 1.645, 95: 1.96, 99: 2.576}.get(int(prediction_interval), 1.96)
        half_width = z * (variance ** 0.5)
        center = -profile.calibration_bias
        confidence = max(0.1, 1.0 - variance * 2)
        return TwinConfidenceLimits(
            twin_id=twin_id,
            prediction_interval_percent=prediction_interval,
            lower_bound=round(max(0.0, center - half_width), 3),
            upper_bound=round(min(1.0, center + half_width), 3),
            confidence_score=round(confidence, 3),
            calibration_version=f"v{profile.sample_count // 100 + 1}",
            calibrated_at=datetime.now(timezone.utc).isoformat(),
        )


class TwinFallbackController:
    def __init__(self) -> None:
        self._modes: dict[str, TwinFallbackMode] = {}

    def evaluate(self, twin_id: str, confidence: TwinConfidenceLimits,
                 threshold: float = 0.5) -> TwinFallbackMode:
        if confidence.confidence_score < threshold:
            mode = "recommendation_only"
            reason = f"Confidence {confidence.confidence_score:.2f} below threshold {threshold}"
        elif confidence.confidence_score < threshold * 0.7:
            mode = "disabled"
            reason = f"Confidence {confidence.confidence_score:.2f} critically low"
        else:
            mode = "full"
            reason = "Confidence acceptable, full twin mode"

        fb = TwinFallbackMode(
            twin_id=twin_id, mode=mode,
            reason=reason, confidence_threshold=threshold,
        )
        self._modes[twin_id] = fb
        return fb

    def get_mode(self, twin_id: str) -> TwinFallbackMode | None:
        return self._modes.get(twin_id)

    def list_modes(self) -> list[TwinFallbackMode]:
        return list(self._modes.values())


class R02TwinCalibrationService:
    def __init__(self) -> None:
        self.calibrator = WorkloadFamilyCalibrator()
        self.confidence = TwinConfidenceCalculator()
        self.fallback = TwinFallbackController()

    def calibrate_for_workload(self, workload: dict, twin_id: str = "default") -> dict:
        family = self.calibrator.get_family(workload)
        profile = self.calibrator.get_profile(family)
        limits = self.confidence.calculate(twin_id, profile)
        mode = self.fallback.evaluate(twin_id, limits)
        return {
            "family": family,
            "profile": profile.model_dump(mode="json"),
            "confidence_limits": limits.model_dump(mode="json"),
            "fallback_mode": mode.model_dump(mode="json"),
            "calibrated_prediction": self.calibrator.calibrate_prediction(family, 0.5),
        }
