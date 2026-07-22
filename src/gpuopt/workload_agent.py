from __future__ import annotations

import logging
import math
import platform
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .gpu_monitor import GPUMonitor
from .schemas import (
    DigitalTwinSimulation,
    JobAssignment,
    MLPredictionResult,
    SystemInfo,
    WorkloadInput,
)

logger = logging.getLogger(__name__)

_HAS_PSUTIL = False
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    pass


class SystemDetector:
    def detect(self, cluster_id: str = "") -> SystemInfo:
        info = SystemInfo(cluster_id=cluster_id)
        try:
            info.hostname = platform.node()
        except Exception:
            pass
        try:
            info.cpu_model = platform.processor() or ""
        except Exception:
            pass
        try:
            info.cpu_cores = (psutil.cpu_count(logical=False) or 0) if _HAS_PSUTIL else 0
            info.cpu_threads = (psutil.cpu_count(logical=True) or 0) if _HAS_PSUTIL else 0
            info.cpu_usage_percent = (psutil.cpu_percent(interval=0.1)) if _HAS_PSUTIL else 0.0
        except Exception:
            pass
        try:
            if _HAS_PSUTIL:
                mem = psutil.virtual_memory()
                info.ram_total_gb = round(mem.total / (1024 ** 3), 1)
                info.ram_available_gb = round(mem.available / (1024 ** 3), 1)
                info.ram_used_gb = round(mem.used / (1024 ** 3), 1)
                info.ram_usage_percent = round(mem.percent, 1)
        except Exception:
            pass
        try:
            monitor = GPUMonitor()
            snap = monitor.collect()
            info.gpu_count = snap.total_gpus
            info.total_gpu_memory_gb = round(snap.total_memory_mb / 1024, 1)
            info.used_gpu_memory_gb = round(snap.used_memory_mb / 1024, 1)
            info.free_gpu_memory_gb = round(snap.free_memory_mb / 1024, 1)
            info.gpus = [
                {
                    "index": d.index,
                    "model": d.model,
                    "memory_total_gb": round(d.memory_total_mb / 1024, 1),
                    "memory_used_gb": round(d.memory_used_mb / 1024, 1),
                    "memory_free_gb": round(d.memory_free_mb / 1024, 1),
                    "utilization_percent": round(d.utilization_gpu_percent, 1),
                    "temperature_celsius": round(d.temperature_celsius, 1),
                    "power_draw_watts": round(d.power_draw_watts, 1),
                }
                for d in snap.devices
            ]
        except Exception as exc:
            logger.debug("GPU detection skipped: %s", exc)
        return info


class MLPredictor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._training_data: list[dict] = []

    def predict(self, workload: WorkloadInput, system: SystemInfo) -> MLPredictionResult:
        with self._lock:
            result = self._run_prediction(workload, system)
            return result

    def train(self, completed_jobs: list[dict]) -> dict:
        with self._lock:
            self._training_data.extend(completed_jobs)
            n = len(self._training_data)
            return {"status": "trained", "samples": n}

    def _run_prediction(self, workload: WorkloadInput, system: SystemInfo) -> MLPredictionResult:
        risk_factors = []
        mem_per_gpu = system.free_gpu_memory_gb / max(system.gpu_count, 1) if system.gpu_count > 0 else 0
        needed = workload.memory_required_gb
        gpu_needed = workload.gpu_required
        cpu_needed = workload.cpu_required_cores

        if system.gpu_count == 0:
            return MLPredictionResult(
                success_probability=0.0,
                risk_factors=["No GPU detected on system"],
                recommendation="Cannot assign job: no GPU available",
            )

        if gpu_needed > system.gpu_count:
            risk_factors.append(f"Need {gpu_needed} GPUs, only {system.gpu_count} available")

        if needed > 0 and mem_per_gpu > 0 and needed > mem_per_gpu:
            risk_factors.append(f"Job needs {needed:.0f}GB/GPU, only {mem_per_gpu:.0f}GB free per GPU")

        if system.ram_available_gb > 0 and cpu_needed > 0 and cpu_needed > system.cpu_threads:
            risk_factors.append(f"Job needs {cpu_needed:.0f} CPU threads, only {system.cpu_threads} available")

        if system.ram_available_gb > 0 and needed > system.ram_available_gb * 0.8:
            risk_factors.append(f"Job memory {needed:.0f}GB may exceed available RAM {system.ram_available_gb:.0f}GB")

        for g in system.gpus:
            if g["temperature_celsius"] > 80:
                risk_factors.append(f"GPU {g['index']} temperature high ({g['temperature_celsius']}C)")
            if g["utilization_percent"] > 90:
                risk_factors.append(f"GPU {g['index']} already at {g['utilization_percent']}% utilization")

        raw_score = self._compute_success_probability(workload, system, risk_factors)
        prob = round(max(0.0, min(1.0, raw_score)), 3)

        predicted_dur = self._estimate_duration(workload, system)
        pred_mem = min(needed * 1.2, system.total_gpu_memory_gb) if needed > 0 else 0
        pred_util = self._estimate_gpu_util(workload, system)

        if prob >= 0.7:
            recommendation = "Job is likely to succeed. Proceeding with assignment."
        elif prob >= 0.4:
            recommendation = "Job may face resource constraints. Consider adjusting requirements."
        else:
            recommendation = "Job is unlikely to succeed with current resources."

        return MLPredictionResult(
            success_probability=prob,
            predicted_duration_minutes=round(predicted_dur, 1),
            predicted_memory_peak_gb=round(pred_mem, 1),
            predicted_gpu_utilization=round(pred_util, 1),
            risk_factors=risk_factors,
            recommendation=recommendation,
        )

    def _compute_success_probability(self, wl: WorkloadInput, sys: SystemInfo, risks: list[str]) -> float:
        score = 0.9
        gpu_ratio = wl.gpu_required / max(sys.gpu_count, 1)
        if gpu_ratio > 0.5:
            score -= 0.2
        elif gpu_ratio > 0.8:
            score -= 0.4
        if wl.memory_required_gb > 0:
            free_per_gpu = sys.free_gpu_memory_gb / max(sys.gpu_count, 1)
            mem_ratio = wl.memory_required_gb / max(free_per_gpu, 1)
            if mem_ratio > 1.0:
                score -= 0.3
            elif mem_ratio > 0.8:
                score -= 0.15
        if sys.ram_usage_percent > 90:
            score -= 0.1
        hot = sum(1 for g in sys.gpus if g.get("temperature_celsius", 0) > 80)
        if hot > 0:
            score -= 0.05 * hot
        num_risks = len(risks)
        score -= 0.05 * num_risks
        if self._training_data:
            n = len(self._training_data)
            recent = self._training_data[-min(n, 50):]
            success_rate = sum(1 for j in recent if j.get("actual_success", False)) / max(len(recent), 1)
            score = score * 0.6 + success_rate * 0.4
        return max(0.05, min(0.99, score))

    def _estimate_duration(self, wl: WorkloadInput, sys: SystemInfo) -> float:
        base = wl.max_duration_minutes
        gpu_power = sum(g.get("power_draw_watts", 300) for g in sys.gpus) / max(sys.gpu_count, 1)
        factor = 300.0 / max(gpu_power, 1)
        mem_factor = 1.0
        if wl.memory_required_gb > 0 and sys.free_gpu_memory_gb > 0:
            mem_factor = wl.memory_required_gb / (sys.free_gpu_memory_gb / max(sys.gpu_count, 1) + 0.1)
        return base * factor * min(mem_factor, 2.0)

    def _estimate_gpu_util(self, wl: WorkloadInput, sys: SystemInfo) -> float:
        base = 75.0
        if wl.framework == "pytorch":
            base = 72.0
        elif wl.framework == "tensorflow":
            base = 68.0
        elif wl.framework == "jax":
            base = 82.0
        if wl.precision in ("bf16", "fp16"):
            base *= 1.15
        if wl.model_size_gb > 0 and sys.total_gpu_memory_gb > 0:
            ratio = wl.model_size_gb / max(sys.total_gpu_memory_gb, 1)
            if ratio > 0.5:
                base *= 0.85
        return min(base, 98.0)


class DigitalTwinSimulator:
    def simulate(self, workload: WorkloadInput, system: SystemInfo, prediction: MLPredictionResult) -> DigitalTwinSimulation:
        sim = DigitalTwinSimulation(
            workload=workload,
            system=system,
            prediction=prediction,
        )
        if not system.gpu_count:
            sim.feasible = False
            sim.rejection_reason = "No GPU available on system"
            return sim
        if workload.gpu_required > system.gpu_count:
            sim.feasible = False
            sim.rejection_reason = f"Need {workload.gpu_required} GPUs, system has {system.gpu_count}"
            return sim
        if prediction.success_probability < 0.4:
            sim.feasible = False
            sim.rejection_reason = "ML prediction indicates low success probability"
            return sim
        if workload.memory_required_gb > 0:
            free_per_gpu = system.free_gpu_memory_gb / max(system.gpu_count, 1)
            if workload.memory_required_gb > free_per_gpu + 5:
                sim.feasible = False
                sim.rejection_reason = (
                    f"Job needs {workload.memory_required_gb:.0f}GB/GPU, "
                    f"only {free_per_gpu:.0f}GB free per GPU"
                )
                return sim
        sim.feasible = True
        assigned = self._select_gpus(workload, system)
        sim.assigned_gpu_indices = assigned
        per_gpu = workload.memory_required_gb if workload.memory_required_gb > 0 else 16.0
        sim.assigned_memory_gb = round(per_gpu * len(assigned), 1)
        gpu_hours = (prediction.predicted_duration_minutes / 60.0) * len(assigned)
        sim.estimated_cost = round(gpu_hours * 2.5, 2)
        return sim

    def _select_gpus(self, workload: WorkloadInput, system: SystemInfo) -> list[int]:
        candidates = sorted(system.gpus, key=lambda g: (g["utilization_percent"], g["temperature_celsius"]))
        return [g["index"] for g in candidates[:workload.gpu_required]]


class WorkloadAgent:
    def __init__(self) -> None:
        self.detector = SystemDetector()
        self.predictor = MLPredictor()
        self.simulator = DigitalTwinSimulator()
        self._assignments: dict[str, JobAssignment] = {}

    def detect_system(self, cluster_id: str = "") -> SystemInfo:
        return self.detector.detect(cluster_id)

    def submit_workload(self, workload: WorkloadInput) -> dict:
        system = self.detect_system(workload.cluster_id)
        prediction = self.predictor.predict(workload, system)
        simulation = self.simulator.simulate(workload, system, prediction)
        if not simulation.feasible:
            return {
                "status": "rejected",
                "workload": workload.model_dump(mode="json"),
                "system": system.model_dump(mode="json"),
                "prediction": prediction.model_dump(mode="json"),
                "simulation": simulation.model_dump(mode="json"),
                "message": f"Workload rejected: {simulation.rejection_reason}",
            }
        assignment_id = str(uuid4())
        assignment = JobAssignment(
            assignment_id=assignment_id,
            workload=workload,
            simulation=simulation,
            assigned_gpu_indices=simulation.assigned_gpu_indices,
            assigned_memory_gb=simulation.assigned_memory_gb,
            assigned_node=system.hostname,
            status="assigned",
        )
        self._assignments[assignment_id] = assignment
        return {
            "status": "assigned",
            "workload": workload.model_dump(mode="json"),
            "system": system.model_dump(mode="json"),
            "prediction": prediction.model_dump(mode="json"),
            "simulation": simulation.model_dump(mode="json"),
            "assignment": assignment.model_dump(mode="json"),
            "message": (
                f"Job '{workload.name}' assigned to {system.hostname} "
                f"on GPUs {assignment.assigned_gpu_indices} "
                f"with {assignment.assigned_memory_gb:.0f}GB memory"
            ),
        }

    def list_assignments(self) -> list[JobAssignment]:
        return sorted(self._assignments.values(), key=lambda a: a.started_at, reverse=True)

    def get_assignment(self, assignment_id: str) -> JobAssignment | None:
        return self._assignments.get(assignment_id)

    def complete_assignment(self, assignment_id: str, success: bool, duration_minutes: float = 0.0) -> JobAssignment | None:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            return None
        assignment.status = "completed" if success else "failed"
        assignment.actual_success = success
        assignment.actual_duration_minutes = duration_minutes
        assignment.completed_at = datetime.now(timezone.utc).isoformat()
        self.predictor.train([{
            "workload": assignment.workload.model_dump(mode="json"),
            "actual_success": success,
            "actual_duration": duration_minutes,
        }])
        return assignment

    def get_stats(self) -> dict:
        total = len(self._assignments)
        completed = sum(1 for a in self._assignments.values() if a.status == "completed")
        failed = sum(1 for a in self._assignments.values() if a.status == "failed")
        assigned = sum(1 for a in self._assignments.values() if a.status == "assigned")
        return {
            "total_assignments": total,
            "assigned": assigned,
            "completed": completed,
            "failed": failed,
            "success_rate": round(completed / max(completed + failed, 1), 3) if completed + failed > 0 else 0.0,
            "ml_training_samples": len(self.predictor._training_data),
        }
