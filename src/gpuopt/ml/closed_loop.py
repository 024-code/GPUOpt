from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .cluster_algorithm import ClusterManagementAlgorithm
from .node_simulation import (
    ClusterTopology,
    EnhancedSimulationEngine,
    SimGpu,
)
from .training_data_pipeline import TrainingDataCollector

logger = logging.getLogger(__name__)


class EpisodeOutcome:
    def __init__(self) -> None:
        self.telemetry_log: list[dict] = []
        self.labels: list[int] = []
        self.decisions: list[dict] = []
        self.faults: list[dict] = []
        self.total_steps: int = 0
        self.total_faults: int = 0
        self.avg_risk: float = 0.0
        self.episode_id: str = uuid.uuid4().hex[:12]

    @property
    def n_samples(self) -> int:
        return len(self.telemetry_log)


class ClosedLoopTrainer:
    def __init__(
        self,
        engine: Any = None,
        management_algorithm: ClusterManagementAlgorithm | None = None,
    ) -> None:
        self._engine = engine
        self._manager = management_algorithm or ClusterManagementAlgorithm()
        self._ml_engine: Any = None
        self._episode_history: list[EpisodeOutcome] = []
        self._training_sessions: int = 0

    @property
    def ml_engine(self) -> Any:
        if self._ml_engine is None:
            try:
                from .engine import MLEngine
                self._ml_engine = MLEngine()
            except Exception:
                pass
        return self._ml_engine

    @property
    def predictor(self) -> Any:
        return self._manager.predictor

    def run_episode(
        self,
        topology: ClusterTopology,
        steps: int = 100,
        profile: dict[str, float] | None = None,
        apply_management: bool = True,
        label_fn: Any = None,
    ) -> EpisodeOutcome:
        sim = EnhancedSimulationEngine(seed=self._training_sessions + 42)
        sim.init_topology(topology)
        outcome = EpisodeOutcome()

        if label_fn is None:
            label_fn = TrainingDataCollector.generate_labels

        for step in range(steps):
            sim.step_simulation(profile=profile)

            snapshot = sim.capture_snapshot()
            metrics = self._manager.get_gpu_metrics(sim.topology)

            for metric in metrics:
                telemetry = metric.to_telemetry()
                risk = self._manager.predict_failure_risk(metric)
                label = 1 if risk > 0.5 else 0
                outcome.telemetry_log.append(telemetry)
                outcome.labels.append(label)

            if apply_management:
                for node in sim.topology.nodes:
                    for gpu in node.gpus:
                        gpu_metrics = next(
                            (m for m in metrics if m.node_id == node.node_id and m.index == gpu.index),
                            None,
                        )
                        if gpu_metrics is None:
                            continue

                        risk = self._manager.predict_failure_risk(gpu_metrics)
                        throttle = self._manager.adaptive_throttle(gpu_metrics, risk)
                        if throttle["action_needed"]:
                            gpu.clock_mhz = throttle["recommended_clock_mhz"]
                            gpu.power_cap_watts = max(
                                gpu.power_cap_watts * 0.5,
                                gpu.power_cap_watts * (1 - throttle["power_reduction_pct"] / 100),
                            )

                actions = self._manager.compute_power_caps(sim.topology)
                for action in actions:
                    for node in sim.topology.nodes:
                        for gpu in node.gpus:
                            if (node.node_id, gpu.index) == action.gpu_key:
                                gpu.power_cap_watts = action.new_cap_watts

                drains = self._manager.recommend_drain(sim.topology)
                for drain in drains:
                    if drain.urgency in ("critical", "high"):
                        for node in sim.topology.nodes:
                            for gpu in node.gpus:
                                if (node.node_id, gpu.index) == drain.gpu_key and not gpu.is_faulted:
                                    gpu.engine_util_pct = 0.0
                                    gpu.memory_used_gib = 0.0

            for event in sim.event_log:
                if event["event_type"] in ("gpu_fault", "cluster_failure"):
                    outcome.faults.append(event)

            snap = sim.capture_snapshot()
            if snap.get("aggregate", {}).get("faulted_gpus", 0) > outcome.total_faults:
                outcome.total_faults = snap["aggregate"]["faulted_gpus"]

        outcome.total_steps = steps
        outcome.avg_risk = float(np.mean(
            [self._manager.predict_failure_risk(m) for m in self._manager.get_gpu_metrics(sim.topology)]
        )) if self._manager.get_gpu_metrics(sim.topology) else 0.0
        self._episode_history.append(outcome)
        logger.info(
            "Episode %s: %d steps, %d samples, %d faults, avg_risk=%.3f",
            outcome.episode_id, steps, outcome.n_samples, outcome.total_faults, outcome.avg_risk,
        )
        return outcome

    def run_curriculum(
        self,
        base_topology: ClusterTopology,
        episodes: int = 5,
        steps_per_episode: int = 100,
        difficulty_escalation: float = 0.2,
    ) -> list[EpisodeOutcome]:
        outcomes: list[EpisodeOutcome] = []
        for ep in range(episodes):
            load = 50.0 + ep * difficulty_escalation * 50.0
            profile = {
                "gpu_util_target": min(98.0, load),
                "memory_target_pct": min(95.0, 40.0 + ep * difficulty_escalation * 30.0),
                "tensor_intensity": min(0.95, 0.5 + ep * difficulty_escalation * 0.1),
                "mem_intensity": min(0.85, 0.4 + ep * difficulty_escalation * 0.1),
            }
            logger.info("Curriculum episode %d/%d: load=%.0f%%", ep + 1, episodes, profile["gpu_util_target"])
            outcome = self.run_episode(base_topology, steps_per_episode, profile)
            outcomes.append(outcome)
        return outcomes

    def retrain_from_episodes(
        self,
        episodes: list[EpisodeOutcome] | None = None,
        blend_with_web: bool = True,
        n_synthetic: int = 500,
    ) -> dict[str, Any]:
        data: list[dict] = []
        labels: list[int] = []

        if episodes is None:
            episodes = self._episode_history

        for ep in episodes:
            data.extend(ep.telemetry_log)
            labels.extend(ep.labels)

        if not data:
            return {"status": "no_data", "message": "No episode data available for retraining"}

        if blend_with_web and self.ml_engine:
            try:
                web_data, web_labels = self.ml_engine.web_ingestion.get_training_data(max_samples=2000)
                data.extend(web_data)
                labels.extend(web_labels)
                logger.info("Blended %d web dataset samples", len(web_data))
            except Exception as exc:
                logger.warning("Web dataset blend skipped: %s", exc)

        result = self.ml_engine.train_ensemble(
            telemetry_history=data, labels=labels, n_synthetic=n_synthetic,
        ) if self.ml_engine else {"status": "no_ml_engine"}

        self._training_sessions += 1
        pos_rate = sum(labels) / max(len(labels), 1)
        result["episode_samples"] = len(data)
        result["positive_rate"] = round(pos_rate, 4)
        result["training_session"] = self._training_sessions
        result["source"] = "closed_loop"

        logger.info(
            "Retrained from %d episode samples (%d positive, %.1f%%)",
            len(data), sum(labels), 100 * pos_rate,
        )
        return result

    def iterative_improvement_cycle(
        self,
        topology: ClusterTopology,
        cycles: int = 3,
        steps_per_episode: int = 80,
        retrain_every: int = 1,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        all_episodes: list[EpisodeOutcome] = []

        for cycle in range(cycles):
            logger.info("Improvement cycle %d/%d", cycle + 1, cycles)
            profile = {
                "gpu_util_target": min(98.0, 60.0 + cycle * 10.0),
                "memory_target_pct": min(95.0, 50.0 + cycle * 10.0),
                "tensor_intensity": min(0.95, 0.6 + cycle * 0.08),
                "mem_intensity": min(0.85, 0.5 + cycle * 0.08),
            }
            outcome = self.run_episode(topology, steps_per_episode, profile)
            all_episodes.append(outcome)

            if (cycle + 1) % retrain_every == 0:
                train_result = self.retrain_from_episodes(all_episodes)
                train_result["cycle"] = cycle + 1
                train_result["faults_this_cycle"] = outcome.total_faults
                results.append(train_result)

            logger.info(
                "Cycle %d: %d faults, avg_risk=%.3f, samples=%d",
                cycle + 1, outcome.total_faults, outcome.avg_risk, outcome.n_samples,
            )

        return results

    def compare_policies(
        self,
        topology: ClusterTopology,
        policies: list[str] | None = None,
        steps: int = 60,
    ) -> list[dict[str, Any]]:
        from .cluster_algorithm import SchedulingPolicy
        if policies is None:
            policies = [p.value for p in SchedulingPolicy]

        results: list[dict[str, Any]] = []
        for policy_name in policies:
            policy = SchedulingPolicy(policy_name)
            self._manager.scheduling_policy = policy
            outcome = self.run_episode(
                topology, steps=steps,
                profile={"gpu_util_target": 80.0, "memory_target_pct": 70.0,
                         "tensor_intensity": 0.7, "mem_intensity": 0.6},
            )
            results.append({
                "policy": policy_name,
                "avg_risk": round(outcome.avg_risk, 4),
                "total_faults": outcome.total_faults,
                "samples_collected": outcome.n_samples,
                "episode_id": outcome.episode_id,
            })
            logger.info("Policy %s: avg_risk=%.4f, faults=%d", policy_name, outcome.avg_risk, outcome.total_faults)
        return results

    def policy_optimization_loop(
        self,
        topology: ClusterTopology,
        iterations: int = 10,
        steps_per_eval: int = 50,
    ) -> dict[str, Any]:
        from .cluster_algorithm import SchedulingPolicy, PowerCapMode

        results: dict[str, Any] = {
            "iterations": [],
            "best_policy": None,
            "best_power_mode": None,
            "best_avg_risk": float("inf"),
        }

        policies = list(SchedulingPolicy)
        cap_modes = list(PowerCapMode)

        for i in range(iterations):
            policy = self._rng_choice(policies)
            cap_mode = self._rng_choice(cap_modes)
            risk_threshold = float(np.random.uniform(0.3, 0.7))
            thermal_threshold = float(np.random.uniform(75, 88))

            self._manager.scheduling_policy = policy
            self._manager.power_cap_mode = cap_mode
            self._manager.risk_threshold = risk_threshold
            self._manager.thermal_threshold_c = thermal_threshold

            outcome = self.run_episode(
                topology, steps=steps_per_eval,
                profile={"gpu_util_target": 85.0, "memory_target_pct": 75.0,
                         "tensor_intensity": 0.8, "mem_intensity": 0.6},
            )

            score = outcome.avg_risk * 0.4 + (outcome.total_faults / max(steps_per_eval, 1)) * 0.6

            iteration_result = {
                "iteration": i + 1,
                "scheduling_policy": policy.value,
                "power_cap_mode": cap_mode.value,
                "risk_threshold": round(risk_threshold, 3),
                "thermal_threshold_c": round(thermal_threshold, 1),
                "avg_risk": round(outcome.avg_risk, 4),
                "total_faults": outcome.total_faults,
                "score": round(score, 4),
            }
            results["iterations"].append(iteration_result)

            if score < results["best_avg_risk"]:
                results["best_avg_risk"] = score
                results["best_policy"] = policy.value
                results["best_power_mode"] = cap_mode.value
                results["best_params"] = {
                    "risk_threshold": round(risk_threshold, 3),
                    "thermal_threshold_c": round(thermal_threshold, 1),
                }

            logger.info(
                "Iteration %d: policy=%s cap=%s risk=%.4f faults=%d score=%.4f",
                i + 1, policy.value, cap_mode.value, outcome.avg_risk, outcome.total_faults, score,
            )

        results["iterations"].sort(key=lambda x: x["score"])
        return results

    def _rng_choice(self, lst: list) -> Any:
        return lst[int(np.random.randint(0, len(lst)))]
