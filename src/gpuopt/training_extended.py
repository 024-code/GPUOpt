from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from .gpu_monitor import GPUMonitor
from .schemas import (
    CheckpointConfig,
    ElasticScalingPlan,
    HPOJob,
    HeterogeneousGpuAssignment,
    QueueAwarePlacement,
)
from .scheduler.rl_scheduler import Node, RLScheduler

logger = logging.getLogger(__name__)


class QueueAwarePlacementEngine:
    def __init__(self) -> None:
        self._rl = RLScheduler(state_size=10, action_size=5)

    def place(self, job: dict, cluster_state: dict | None = None) -> QueueAwarePlacement:
        job_gpus = job.get("gpu_required", 1)
        job_mem = job.get("memory_required_gb", 16)
        priority = job.get("priority", 5)
        job_id = job.get("job_id", f"job-{random.randint(1000, 9999)}")

        try:
            monitor = GPUMonitor()
            snap = monitor.collect()
            nodes_data = [
                {"id": f"node-{d.index // 4}", "gpus": d.index % 4 + 1,
                 "free_mem": (d.memory_total_mb - d.memory_used_mb) / 1024,
                 "model": d.model, "util": d.utilization_gpu_percent}
                for i, d in enumerate(snap.devices)
            ]
        except Exception:
            nodes_data = [
                {"id": f"node-{i}", "gpus": random.randint(1, 4),
                 "free_mem": random.uniform(10, 80), "model": random.choice(["H100", "A100"]),
                 "util": random.uniform(10, 90)}
                for i in range(random.randint(2, 8))
            ]

        best_node = self._find_best_node({"gpu_required": job_gpus, "memory_required_gb": job_mem}, nodes_data)
        queue_wait = self._estimate_queue_wait(len(nodes_data), priority)
        preemptible = priority < 5
        checkpoint = preemptible and random.random() < 0.6

        return QueueAwarePlacement(
            job_id=job_id,
            placement_node=best_node,
            placement_gpus=list(range(job_gpus)),
            queue_wait_predicted_minutes=round(queue_wait, 1),
            priority=priority,
            preemptible=preemptible,
            checkpoint_available=checkpoint,
            estimated_duration_minutes=round(job.get("max_duration_minutes", 120) * random.uniform(0.5, 1.5), 1),
        )

    def _find_best_node(self, job: dict, nodes: list[dict]) -> str:
        candidates = [
            n for n in nodes
            if n["free_mem"] >= job.get("memory_required_gb", 16)
        ]
        if not candidates:
            candidates = sorted(nodes, key=lambda n: n["free_mem"], reverse=True)
        candidates.sort(key=lambda n: (n["util"], -n["free_mem"]))
        return candidates[0]["id"] if candidates else "unknown"

    def _estimate_queue_wait(self, queue_depth: int, priority: int) -> float:
        base = queue_depth * random.uniform(1, 5)
        priority_factor = max(0.2, 1.0 - priority / 10 * 0.8)
        return base * priority_factor


class ElasticScalingPlanner:
    def plan_scale(self, job_id: str = "", workload: dict | None = None,
                   current_workers: int = 1, cluster_load: float = 0.5) -> ElasticScalingPlan:
        wl = workload or {}
        efficiency = self._compute_parallel_efficiency(wl)
        speedup_gain = efficiency * (1 - cluster_load)
        target = current_workers

        if cluster_load < 0.7 and efficiency > 0.5:
            target = min(64, int(current_workers * (1 + speedup_gain)))
        elif cluster_load > 0.85 or efficiency < 0.2:
            target = max(1, int(current_workers * 0.5))

        speedup = self._compute_speedup(current_workers, target, efficiency)
        cost_impact = (target - current_workers) * wl.get("gpu_required", 1) * 0.85

        return ElasticScalingPlan(
            job_id=job_id,
            current_workers=current_workers,
            target_workers=target,
            scaling_reason="Load-based elastic scaling",
            estimated_speedup=round(speedup, 2),
            estimated_cost_impact=round(cost_impact, 2),
            min_workers=1,
            max_workers=64,
        )

    def _compute_speedup(self, current: int, target: int, efficiency: float) -> float:
        if target == current:
            return 1.0
        P = efficiency
        N_ratio = target / max(current, 1)
        return 1.0 / ((1 - P) + P / N_ratio)

    def _compute_parallel_efficiency(self, job: dict) -> float:
        return {"pytorch": 0.7, "tensorflow": 0.5, "jax": 0.85}.get(job.get("framework", "pytorch"), 0.6)


class CheckpointManager:
    def configure(self, job_id: str = "", job_params: dict | None = None) -> CheckpointConfig:
        params = job_params or {}
        duration = params.get("max_duration_minutes", 120)
        model_size = params.get("model_size_gb", 1)
        interval = max(5, min(120, int(duration / 10)))
        ckpt_size = model_size * 2
        restore = ckpt_size * 2

        return CheckpointConfig(
            checkpoint_enabled=True,
            checkpoint_interval_minutes=interval,
            checkpoint_path=f"/checkpoints/{job_id}/",
            last_checkpoint_time="",
            checkpoint_size_gb=round(ckpt_size, 1),
            restore_time_seconds=round(restore, 0),
        )

    def should_preempt(self, job: dict, checkpoint_config: CheckpointConfig, urgency: str = "medium") -> bool:
        if not checkpoint_config.checkpoint_enabled:
            return False
        if urgency == "critical":
            return True
        if urgency == "high":
            return checkpoint_config.last_checkpoint_time != ""
        return random.random() < 0.3

    def estimate_checkpoint_overhead(self, model_size_gb: float, interval_minutes: int) -> float:
        write_time = model_size_gb * 2 / 60
        overhead = write_time / interval_minutes * 100
        return round(min(overhead, 20), 1)


class HeterogeneousGpuAssigner:
    def assign(self, job_id: str = "", workload: dict | None = None,
               available_gpus: list[dict] | None = None) -> HeterogeneousGpuAssignment:
        wl = workload or {}
        if not available_gpus:
            try:
                monitor = GPUMonitor()
                snap = monitor.collect()
                available_gpus = [{"model": d.model, "memory_mb": d.memory_total_mb,
                                   "compute_cap": "9.0" if "H100" in d.model else "8.0"}
                                  for d in snap.devices]
            except Exception:
                available_gpus = [
                    {"model": "H100", "memory_mb": 81920, "compute_cap": "9.0"},
                    {"model": "A100", "memory_mb": 81920, "compute_cap": "8.0"},
                    {"model": "A100", "memory_mb": 40960, "compute_cap": "8.0"},
                ]

        models = list(set(g["model"] for g in available_gpus))
        primary = models[0] if models else "H100"
        secondary = models[1] if len(models) > 1 else primary
        p_count = sum(1 for g in available_gpus if g["model"] == primary)
        s_count = sum(1 for g in available_gpus if g["model"] == secondary)

        compat = self._compute_compatibility(
            {"model": primary, "compute_cap": "9.0"},
            {"model": secondary, "compute_cap": "8.0"},
        )
        strategy = "uniform" if primary == secondary else "pipeline" if compat > 0.5 else "mixed"

        speedup = 1.0
        if strategy == "pipeline":
            speedup = 1.0 + (1.0 - p_count / max(p_count + s_count, 1)) * 0.3
        elif strategy == "mixed":
            speedup = 1.0 + (compat - 0.5) * 0.2

        return HeterogeneousGpuAssignment(
            job_id=job_id,
            primary_gpu_model=primary,
            secondary_gpu_model=secondary,
            primary_gpu_count=p_count,
            secondary_gpu_count=s_count,
            strategy=strategy,
            expected_speedup=round(speedup, 2),
            compatibility_score=round(compat, 3),
        )

    def _compute_compatibility(self, gpu1: dict, gpu2: dict) -> float:
        cc1 = float(gpu1.get("compute_cap", "8.0"))
        cc2 = float(gpu2.get("compute_cap", "8.0"))
        compat = 1.0 - abs(cc1 - cc2) * 0.3
        if gpu1["model"] == gpu2["model"]:
            compat = 1.0
        return max(0.1, compat)


class HPOManager:
    def __init__(self) -> None:
        self._jobs: dict[str, HPOJob] = {}

    def create_job(self, search_algorithm: str = "bayesian", max_trials: int = 100,
                   parallel_trials: int = 4) -> HPOJob:
        job_id = f"hpo-{random.randint(10000, 99999)}"
        job = HPOJob(
            job_id=job_id,
            search_algorithm=search_algorithm,
            max_trials=max_trials,
            parallel_trials=parallel_trials,
            status="running",
        )
        self._jobs[job_id] = job
        return job

    def submit_trial(self, job_id: str, hyperparameters: dict) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}
        trial_id = f"trial-{len(job.trials) + 1}"
        trial = {"trial_id": trial_id, "hyperparameters": hyperparameters, "status": "running"}
        job.trials.append(trial)
        return trial

    def complete_trial(self, job_id: str, trial_id: str, score: float) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}
        for t in job.trials:
            if t.get("trial_id") == trial_id:
                t["status"] = "completed"
                t["score"] = score
                if job.best_trial_id == "" or score < job.best_score:
                    job.best_score = score
                    job.best_trial_id = trial_id
                return {"status": "completed", "trial_id": trial_id, "score": score}
        return {"error": "Trial not found"}

    def get_best_config(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job or not job.best_trial_id:
            return {"error": "No best config found"}
        for t in job.trials:
            if t.get("trial_id") == job.best_trial_id:
                return {"best_hyperparameters": t["hyperparameters"], "best_score": job.best_score}
        return {"error": "Best trial not found"}


class ExtendedTrainingService:
    def __init__(self) -> None:
        self._placer = QueueAwarePlacementEngine()
        self._scaler = ElasticScalingPlanner()
        self._ckpt = CheckpointManager()
        self._assigner = HeterogeneousGpuAssigner()
        self._hpo = HPOManager()

    def submit_training_job(self, job: dict) -> dict:
        placement = self._placer.place(job)
        scaling = self._scaler.plan_scale(job_id=placement.job_id, workload=job)
        ckpt = self._ckpt.configure(job_id=placement.job_id, job_params=job)
        hetero = self._assigner.assign(job_id=placement.job_id, workload=job)
        hpo = self._hpo.create_job()

        return {
            "status": "planned",
            "placement": placement.model_dump(mode="json"),
            "scaling": scaling.model_dump(mode="json"),
            "checkpoint": ckpt.model_dump(mode="json"),
            "heterogeneous_assignment": hetero.model_dump(mode="json"),
            "hpo": hpo.model_dump(mode="json"),
            "message": f"Training job '{job.get('name', 'unknown')}' planned: "
                       f"place on {placement.placement_node}, "
                       f"scale {scaling.current_workers}->{scaling.target_workers} workers, "
                       f"{hetero.strategy} GPU strategy",
        }
