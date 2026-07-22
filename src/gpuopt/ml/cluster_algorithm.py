from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RISK_AWARE = "risk_aware"
    THERMAL_AWARE = "thermal_aware"
    POWER_EFFICIENT = "power_efficient"
    HYBRID = "hybrid"


class PowerCapMode(Enum):
    OFF = "off"
    TEMPERATURE_GUIDED = "temperature_guided"
    RISK_GUIDED = "risk_guided"
    PREDICTIVE = "predictive"


@dataclass
class GpuMetrics:
    index: int
    node_id: str
    engine_util_pct: float
    memory_pct: float
    gpu_temp_c: float
    power_watts: float
    power_cap_watts: float
    xid_errors: int
    ecc_errors: int
    fan_speed_pct: float
    clock_mhz: float
    wear_factor: float

    def to_telemetry(self) -> dict:
        return {
            "gpu_utilization": self.engine_util_pct,
            "memory_utilization": self.memory_pct,
            "temperature": self.gpu_temp_c,
            "power_usage": self.power_watts,
            "clock_speed": self.clock_mhz,
            "ecc_errors": self.ecc_errors,
            "xid_errors": self.xid_errors,
            "available_gpus": 0,
            "total_gpus": 1,
            "queue_length": 0,
            "job_failures": 0,
            "job_retries": 0,
            "average_job_duration": 0.0,
            "utilization_variance": 0.0,
            "temperature_variance": 0.0,
            "retired_pages": 0,
        }


@dataclass
class JobSpec:
    job_id: str = ""
    name: str = ""
    required_gpus: int = 1
    required_memory_gib: float = 8.0
    estimated_runtime_hours: float = 1.0
    priority: int = 5
    workload_type: str = "llm_inference"
    tensor_intensity: float = 0.6
    mem_intensity: float = 0.5


@dataclass
class ScheduleDecision:
    job_id: str
    assigned_gpus: list[tuple[str, int]]
    policy: SchedulingPolicy
    predicted_failure_risk: float = 0.0
    estimated_power_watts: float = 0.0
    thermal_headroom_c: float = 0.0
    score: float = 0.0
    rationale: str = ""


@dataclass
class CapAction:
    gpu_key: tuple[str, int]
    current_power_watts: float
    new_cap_watts: float
    reason: str
    temp_before: float
    temp_after_estimate: float


@dataclass
class DrainRecommendation:
    gpu_key: tuple[str, int]
    risk_score: float
    reason: str
    suggested_action: str
    urgency: str


class ClusterManagementAlgorithm:
    def __init__(self, predictor: Any = None) -> None:
        self._predictor = predictor
        self.scheduling_policy: SchedulingPolicy = SchedulingPolicy.HYBRID
        self.power_cap_mode: PowerCapMode = PowerCapMode.PREDICTIVE
        self.risk_threshold: float = 0.5
        self.thermal_threshold_c: float = 82.0
        self.power_headroom_ratio: float = 0.15
        self.degradation_warn_threshold: float = 0.15
        self.degradation_drain_threshold: float = 0.30
        self._decision_log: list[dict] = []
        self._cap_log: list[dict] = []
        self._drain_log: list[dict] = []

    @property
    def predictor(self) -> Any:
        if self._predictor is None:
            try:
                from ..predictor.ensemble_failure_predictor import EnsembleFailurePredictor
                self._predictor = EnsembleFailurePredictor()
            except Exception:
                self._predictor = None
        return self._predictor

    def get_gpu_metrics(
        self, topology: Any,
    ) -> list[GpuMetrics]:
        metrics: list[GpuMetrics] = []
        for node in topology.nodes:
            if not node.is_on:
                continue
            for gpu in node.gpus:
                if gpu.is_faulted:
                    continue
                metrics.append(GpuMetrics(
                    index=gpu.index, node_id=node.node_id,
                    engine_util_pct=gpu.engine_util_pct,
                    memory_pct=gpu.memory_pct,
                    gpu_temp_c=gpu.gpu_temp_c,
                    power_watts=gpu.power_draw_watts,
                    power_cap_watts=gpu.power_cap_watts,
                    xid_errors=gpu.xid_errors,
                    ecc_errors=gpu.ecc_corrected + gpu.ecc_uncorrected,
                    fan_speed_pct=gpu.fan_speed_pct,
                    clock_mhz=gpu.clock_mhz,
                    wear_factor=gpu.degradation.wear_factor if hasattr(gpu, "degradation") else 0.0,
                ))
        return metrics

    def predict_failure_risk(self, metric: GpuMetrics) -> float:
        if self.predictor is None:
            return self._heuristic_risk(metric)
        try:
            result = self.predictor.predict_failure(metric.to_telemetry())
            return result.get("probability_raw", result.get("probability", 0.0))
        except Exception:
            return self._heuristic_risk(metric)

    def _heuristic_risk(self, metric: GpuMetrics) -> float:
        risk = 0.0
        risk += max(0, metric.gpu_temp_c - 70) / 30.0 * 0.25
        risk += (metric.engine_util_pct / 100.0) * 0.15
        risk += (metric.memory_pct / 100.0) * 0.15
        risk += min(1.0, metric.xid_errors / 10.0) * 0.15
        risk += min(1.0, metric.ecc_errors / 20.0) * 0.10
        risk += metric.wear_factor * 0.10
        risk += max(0, metric.power_watts / metric.power_cap_watts - 0.8) * 0.10
        return min(1.0, risk)

    def _compute_gpu_score(
        self, metric: GpuMetrics, job: JobSpec, policy: SchedulingPolicy,
    ) -> dict:
        risk = self.predict_failure_risk(metric)
        temp_ratio = metric.gpu_temp_c / self.thermal_threshold_c
        mem_free_pct = 1.0 - metric.memory_pct / 100.0
        util_free = 1.0 - metric.engine_util_pct / 100.0
        power_efficiency = metric.engine_util_pct / max(metric.power_watts, 1)
        wear_penalty = metric.wear_factor

        scores = {}
        if policy == SchedulingPolicy.ROUND_ROBIN or policy == SchedulingPolicy.HYBRID:
            scores["load"] = 1.0 - metric.engine_util_pct / 100.0
        if policy == SchedulingPolicy.LEAST_LOADED or policy == SchedulingPolicy.HYBRID:
            scores["load"] = 1.0 - metric.engine_util_pct / 100.0
        if policy == SchedulingPolicy.RISK_AWARE or policy == SchedulingPolicy.HYBRID:
            scores["safety"] = 1.0 - risk
        if policy == SchedulingPolicy.THERMAL_AWARE or policy == SchedulingPolicy.HYBRID:
            scores["thermal"] = max(0.0, 1.0 - temp_ratio)
        if policy == SchedulingPolicy.POWER_EFFICIENT or policy == SchedulingPolicy.HYBRID:
            scores["power"] = min(1.0, power_efficiency * 0.5)

        scores["memory"] = mem_free_pct
        scores["availability"] = util_free

        weights = {"safety": 0.30, "thermal": 0.20, "load": 0.15, "memory": 0.15, "availability": 0.10, "power": 0.10}
        final_score = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights)
        final_score *= (1.0 - wear_penalty * 0.5)

        return {
            "risk": round(risk, 4),
            "raw_score": round(final_score, 4),
            "scores": {k: round(v, 4) for k, v in scores.items() if k in scores},
            "temp_ratio": round(temp_ratio, 3),
            "mem_free_pct": round(mem_free_pct * 100, 1),
        }

    def schedule_job(
        self,
        job: JobSpec,
        topology: Any,
        policy: SchedulingPolicy | None = None,
    ) -> ScheduleDecision:
        if policy:
            self.scheduling_policy = policy

        metrics = self.get_gpu_metrics(topology)
        if len(metrics) < job.required_gpus:
            return ScheduleDecision(
                job_id=job.job_id, assigned_gpus=[], policy=self.scheduling_policy,
                score=0.0, rationale=f"Insufficient GPUs: need {job.required_gpus}, have {len(metrics)}",
            )

        scored_gpus = [
            (m, self._compute_gpu_score(m, job, self.scheduling_policy))
            for m in metrics
            if m.memory_pct <= 100 - (job.required_memory_gib / max(1, m.power_cap_watts) * 100)
        ]

        if not scored_gpus:
            scored_gpus = [(m, self._compute_gpu_score(m, job, self.scheduling_policy)) for m in metrics]

        scored_gpus.sort(key=lambda x: x[1]["raw_score"], reverse=True)
        selected = scored_gpus[: job.required_gpus]

        avg_risk = sum(s[1]["risk"] for s in selected) / max(len(selected), 1)
        total_power = sum(s[0].power_watts for s in selected)
        avg_temp = sum(s[0].gpu_temp_c for s in selected)

        assigned = [(s[0].node_id, s[0].index) for s in selected]
        best_score = selected[0][1]["raw_score"] if selected else 0.0

        risk_level = "high" if avg_risk > self.risk_threshold else "medium" if avg_risk > self.risk_threshold * 0.6 else "low"
        rationale = (
            f"Scheduled {len(selected)} GPUs via {self.scheduling_policy.value} | "
            f"Avg risk: {avg_risk:.2f} ({risk_level}) | "
            f"Top GPU score: {best_score:.3f} | "
            f"Est. power: {total_power:.0f}W | "
            f"Avg temp: {avg_temp:.0f}C"
        )

        decision = ScheduleDecision(
            job_id=job.job_id, assigned_gpus=assigned,
            policy=self.scheduling_policy, predicted_failure_risk=avg_risk,
            estimated_power_watts=total_power, thermal_headroom_c=self.thermal_threshold_c - avg_temp,
            score=best_score, rationale=rationale,
        )

        self._decision_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job.job_id, "policy": self.scheduling_policy.value,
            "assigned_gpus": assigned, "avg_risk": avg_risk, "score": best_score,
        })
        return decision

    def compute_power_caps(
        self,
        topology: Any,
        mode: PowerCapMode | None = None,
    ) -> list[CapAction]:
        if mode:
            self.power_cap_mode = mode

        actions: list[CapAction] = []
        metrics = self.get_gpu_metrics(topology)

        for metric in metrics:
            risk = self.predict_failure_risk(metric)
            temp_margin = self.thermal_threshold_c - metric.gpu_temp_c
            current_cap = metric.power_cap_watts
            new_cap = current_cap
            reason = ""

            if self.power_cap_mode == PowerCapMode.TEMPERATURE_GUIDED:
                if metric.gpu_temp_c > self.thermal_threshold_c - 5:
                    reduction = min(100, (metric.gpu_temp_c - self.thermal_threshold_c + 5) * 15)
                    new_cap = max(current_cap * 0.5, current_cap - reduction)
                    reason = f"Temp {metric.gpu_temp_c:.0f}C near threshold"
                elif metric.gpu_temp_c < 60:
                    new_cap = min(metric.power_cap_watts, current_cap + 20)
                    reason = "Low temp, increasing cap"

            elif self.power_cap_mode == PowerCapMode.RISK_GUIDED:
                if risk > self.risk_threshold:
                    reduction = min(200, risk * 300)
                    new_cap = max(current_cap * 0.4, current_cap - reduction)
                    reason = f"Risk {risk:.2f} > threshold {self.risk_threshold}"
                elif risk < self.risk_threshold * 0.5:
                    new_cap = metric.power_cap_watts
                    reason = "Low risk, full cap"

            elif self.power_cap_mode == PowerCapMode.PREDICTIVE:
                risk = self.predict_failure_risk(metric)
                temp_ratio = metric.gpu_temp_c / self.thermal_threshold_c
                combined = risk * 0.5 + temp_ratio * 0.5

                if combined > 0.6:
                    reduction = min(250, combined * 350)
                    new_cap = max(current_cap * 0.35, current_cap - reduction)
                    reason = f"Predictive: risk={risk:.2f}, temp_ratio={temp_ratio:.2f}"
                elif combined < 0.3 and metric.gpu_temp_c < 65:
                    new_cap = metric.power_cap_watts
                    reason = "Predictive: safe envelope"

            if new_cap != current_cap:
                temp_drop = max(0, (current_cap - new_cap) / current_cap) * 8
                actions.append(CapAction(
                    gpu_key=(metric.node_id, metric.index),
                    current_power_watts=metric.power_watts,
                    new_cap_watts=round(new_cap, 0),
                    reason=reason,
                    temp_before=metric.gpu_temp_c,
                    temp_after_estimate=round(metric.gpu_temp_c - temp_drop, 1),
                ))

        self._cap_log.extend([
            {"timestamp": datetime.now(timezone.utc).isoformat(),
             "gpu_key": str(a.gpu_key), "from": a.current_power_watts,
             "to": a.new_cap_watts, "reason": a.reason}
            for a in actions
        ])
        return actions

    def recommend_drain(
        self,
        topology: Any,
    ) -> list[DrainRecommendation]:
        recommendations: list[DrainRecommendation] = []
        metrics = self.get_gpu_metrics(topology)

        for metric in metrics:
            risk = self.predict_failure_risk(metric)
            wear = metric.wear_factor
            xid_rate = metric.xid_errors
            ecc_rate = metric.ecc_errors
            temp = metric.gpu_temp_c

            reasons: list[str] = []
            urgency = "low"

            if risk > self.risk_threshold:
                reasons.append(f"Failure risk {risk:.2f} exceeds threshold")
                urgency = "critical"

            if wear > self.degradation_drain_threshold:
                reasons.append(f"Wear factor {wear:.3f} exceeds drain threshold")
                urgency = max(urgency, "high")

            if xid_rate > 5:
                reasons.append(f"XID errors elevated ({xid_rate})")
                urgency = max(urgency, "high")

            if ecc_rate > 50:
                reasons.append(f"ECC errors elevated ({ecc_rate})")
                urgency = max(urgency, "medium")

            if temp > self.thermal_threshold_c:
                reasons.append(f"Temperature {temp:.0f}C above threshold")
                urgency = max(urgency, "medium")

            if wear > self.degradation_warn_threshold and wear <= self.degradation_drain_threshold:
                reasons.append(f"Wear factor {wear:.3f} entering warning zone")
                urgency = max(urgency, "low")

            if reasons:
                action = "drain_immediately" if urgency in ("critical", "high") else "monitor_closely"
                recommendations.append(DrainRecommendation(
                    gpu_key=(metric.node_id, metric.index),
                    risk_score=round(risk, 4),
                    reason="; ".join(reasons),
                    suggested_action=action,
                    urgency=urgency,
                ))

        recommendations.sort(key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}[r.urgency])
        self._drain_log.extend([
            {"timestamp": datetime.now(timezone.utc).isoformat(),
             "gpu_key": str(r.gpu_key), "risk": r.risk_score,
             "urgency": r.urgency, "action": r.suggested_action}
            for r in recommendations
        ])
        return recommendations

    def adaptive_throttle(
        self, metric: GpuMetrics, risk: float | None = None,
    ) -> dict[str, Any]:
        if risk is None:
            risk = self.predict_failure_risk(metric)

        clock_reduction = 0.0
        power_reduction = 0.0
        reason = ""

        if risk > self.risk_threshold:
            severity = (risk - self.risk_threshold) / (1.0 - self.risk_threshold)
            clock_reduction = min(0.5, severity * 0.5)
            power_reduction = min(0.4, severity * 0.4)
            reason = f"Risk-based: risk={risk:.2f}, severity={severity:.2f}"
        elif metric.gpu_temp_c > self.thermal_threshold_c:
            severity = (metric.gpu_temp_c - self.thermal_threshold_c) / 15.0
            clock_reduction = min(0.3, severity * 0.3)
            power_reduction = min(0.25, severity * 0.25)
            reason = f"Thermal: temp={metric.gpu_temp_c:.0f}C"

        return {
            "risk": round(risk, 4),
            "clock_reduction_pct": round(clock_reduction * 100, 1),
            "power_reduction_pct": round(power_reduction * 100, 1),
            "recommended_clock_mhz": round(metric.clock_mhz * (1.0 - clock_reduction), 0),
            "reason": reason,
            "action_needed": clock_reduction > 0.05,
        }

    def balance_load(
        self, topology: Any, threshold_imbalance: float = 0.25,
    ) -> list[dict[str, Any]]:
        metrics = self.get_gpu_metrics(topology)
        if not metrics:
            return []

        utils = [m.engine_util_pct for m in metrics]
        mean_util = np.mean(utils)
        std_util = np.std(utils) if len(utils) > 1 else 0.0

        if std_util / max(mean_util, 1) < threshold_imbalance:
            return [{"balanced": True, "mean_util": round(mean_util, 1), "std_util": round(std_util, 1)}]

        overloaded = [m for m in metrics if m.engine_util_pct > mean_util + std_util * 1.5]
        underloaded = [m for m in metrics if m.engine_util_pct < mean_util - std_util * 0.5]

        actions: list[dict[str, Any]] = []
        for src in overloaded:
            for dst in underloaded:
                if len(actions) >= sum(1 for _ in overloaded):
                    break
                drift = src.engine_util_pct - dst.engine_util_pct
                if drift > 20:
                    actions.append({
                        "from": (src.node_id, src.index),
                        "to": (dst.node_id, dst.index),
                        "drift_pct": round(drift, 1),
                        "src_util": src.engine_util_pct,
                        "dst_util": dst.engine_util_pct,
                        "action": f"Migrate load from {src.node_id}:GPU{src.index} to {dst.node_id}:GPU{dst.index}",
                    })

        return actions if actions else [{"balanced": True, "mean_util": round(mean_util, 1), "std_util": round(std_util, 1)}]

    def get_cluster_health_report(self, topology: Any) -> dict[str, Any]:
        metrics = self.get_gpu_metrics(topology)
        if not metrics:
            return {"error": "no healthy GPUs found"}

        risks = [self.predict_failure_risk(m) for m in metrics]
        caps = self.compute_power_caps(topology)
        drains = self.recommend_drain(topology)
        throttle_actions = [self.adaptive_throttle(m, r) for m, r in zip(metrics, risks)]
        balance = self.balance_load(topology)

        avg_util = np.mean([m.engine_util_pct for m in metrics])
        avg_temp = np.mean([m.gpu_temp_c for m in metrics])
        max_temp = max(m.gpu_temp_c for m in metrics)
        total_power = sum(m.power_watts for m in metrics)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cluster_health": {
                "gpu_count": len(metrics),
                "avg_util_pct": round(avg_util, 1),
                "avg_temp_c": round(avg_temp, 1),
                "max_temp_c": round(max_temp, 1),
                "total_power_watts": round(total_power, 1),
                "avg_risk": round(np.mean(risks), 4),
                "max_risk": round(max(risks), 4),
                "gpus_at_risk": sum(1 for r in risks if r > self.risk_threshold),
                "gpus_warning": sum(1 for r in risks if self.risk_threshold * 0.6 < r <= self.risk_threshold),
            },
            "recommendations": {
                "power_cap_actions": [
                    {"gpu": str(a.gpu_key), "from_watts": a.current_power_watts,
                     "to_watts": a.new_cap_watts, "reason": a.reason}
                    for a in caps[:10]
                ],
                "drain_recommendations": [
                    {"gpu": str(d.gpu_key), "risk": d.risk_score,
                     "urgency": d.urgency, "action": d.suggested_action, "reason": d.reason}
                    for d in drains
                ],
                "throttle_actions": [
                    {"gpu": f"{m.node_id}:GPU{m.index}", "risk": t["risk"],
                     "clock_reduction_pct": t["clock_reduction_pct"],
                     "recommended_clock_mhz": t["recommended_clock_mhz"],
                     "reason": t["reason"]}
                    for m, t in zip(metrics, throttle_actions) if t["action_needed"]
                ],
                "load_balance_actions": balance,
            },
            "policies": {
                "scheduling": self.scheduling_policy.value,
                "power_cap_mode": self.power_cap_mode.value,
                "risk_threshold": self.risk_threshold,
                "thermal_threshold_c": self.thermal_threshold_c,
                "degradation_drain_threshold": self.degradation_drain_threshold,
            },
        }
