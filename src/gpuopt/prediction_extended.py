from __future__ import annotations

import logging
import math
import random
from collections import deque
from typing import Any

import numpy as np

from .gpu_monitor import GPUMonitor
from .ml.forecast_model import ForecastModel
from .schemas import (
    ActionImpactForecast,
    ComprehensivePrediction,
    DemandBurstDetection,
    JCTPrediction,
    OOMRiskPrediction,
    QueuePressureForecast,
    ThermalRiskPrediction,
)

logger = logging.getLogger(__name__)


class QueuePressurePredictor:
    def __init__(self) -> None:
        self._history: list[float] = []
        self._alpha = 0.3

    def predict(self, queue_telemetry: list[dict] | None = None, horizon_hours: float = 1.0) -> QueuePressureForecast:
        depth = random.randint(0, 50)
        self._history.append(depth)
        if len(self._history) > 100:
            self._history.pop(0)

        if len(self._history) >= 5:
            xs = np.arange(len(self._history))
            ys = np.array(self._history)
            coeffs = np.polyfit(xs, ys, 1)
            trend = coeffs[0]
            pred_depth = max(0, depth + trend * horizon_hours * 12)
        else:
            trend = 0
            pred_depth = float(depth)

        congestion = 1.0 / (1.0 + math.exp(-(pred_depth - 25) / 8))
        wait_min = pred_depth * random.uniform(0.5, 2.0)

        if congestion > 0.7:
            level = "critical"
            actions = ["Scale up worker nodes", "Preempt low-priority jobs", "Increase queue capacity"]
        elif congestion > 0.4:
            level = "high"
            actions = ["Consider scaling up", "Prioritize short jobs"]
        elif congestion > 0.2:
            level = "medium"
            actions = ["Monitor queue growth"]
        else:
            level = "low"
            actions = ["No action needed"]

        return QueuePressureForecast(
            forecast_horizon_hours=horizon_hours,
            current_queue_depth=depth,
            predicted_queue_depth=round(pred_depth, 1),
            predicted_wait_time_minutes=round(wait_min, 1),
            congestion_probability=round(congestion, 3),
            pressure_level=level,
            recommended_actions=actions,
        )

    def train(self, historical_data: list[dict]) -> None:
        for entry in historical_data:
            if "queue_depth" in entry:
                self._history.append(entry["queue_depth"])


class JCTPredictor:
    def __init__(self) -> None:
        self._completed: list[dict] = []

    def predict(self, job: dict, cluster_state: dict | None = None) -> JCTPrediction:
        gpus = job.get("gpu_required", 1)
        memory = job.get("memory_required_gb", 16)
        framework = job.get("framework", "pytorch")
        dataset = job.get("dataset_size_gb", 10)
        model_size = job.get("model_size_gb", 1)

        base = 120.0
        base += gpus * 5
        base += memory * 0.5
        base += dataset * 0.3
        base += model_size * 10
        if framework == "tensorflow":
            base *= 0.9
        elif framework == "jax":
            base *= 0.8

        noise = random.gauss(0, base * 0.1)
        estimated = max(1.0, base + noise)
        p50 = estimated * 0.9
        p95 = estimated * 1.8
        p99 = estimated * 2.5

        factor_msgs = []
        if gpus > 4:
            factor_msgs.append("Multi-GPU training may have communication overhead")
        if dataset > 100:
            factor_msgs.append("Large dataset may increase I/O wait")
        if model_size > 10:
            factor_msgs.append("Large model may increase computation time")

        confidence = 0.7
        if len(self._completed) >= 10:
            recent = [c.get("duration", estimated) for c in self._completed[-50:]]
            mae = np.mean([abs(d - estimated) for d in recent])
            confidence = max(0.3, 1.0 - mae / max(estimated, 1))

        return JCTPrediction(
            job_id=job.get("job_id", ""),
            estimated_duration_minutes=round(estimated, 1),
            p50_duration_minutes=round(p50, 1),
            p95_duration_minutes=round(p95, 1),
            p99_duration_minutes=round(p99, 1),
            confidence=round(confidence, 2),
            factors=factor_msgs,
        )

    def train(self, completed_jobs: list[dict]) -> None:
        self._completed.extend(completed_jobs)


class OOMRiskPredictor:
    def predict(self, job: dict, gpu_snapshot: dict) -> OOMRiskPrediction:
        job_gpu = job.get("gpu_index", 0)
        mem_required = job.get("memory_required_gb", 16)
        batch_size = job.get("batch_size", 32)
        model_size = job.get("model_size_gb", 1)
        precision = job.get("precision", "fp32")

        precision_factor = {"fp32": 1.0, "fp16": 0.5, "bf16": 0.5, "int8": 0.25}.get(precision, 1.0)
        peak_mem = (model_size * 2 + batch_size * 0.01 + mem_required) * precision_factor

        devices = gpu_snapshot.get("devices", [])
        avail_mem = 0.0
        for d in devices:
            if d.get("index") == job_gpu:
                avail_mem = (d.get("memory_total_mb", 0) - d.get("memory_used_mb", 0)) / 1024
                break
        if avail_mem == 0 and devices:
            avail_mem = max((d.get("memory_total_mb", 81920) - d.get("memory_used_mb", 40960)) / 1024
                           for d in devices)

        if avail_mem > 0:
            prob = peak_mem / avail_mem
        else:
            prob = 0.5

        if prob > 0.9:
            level = "high"
            rec = "Reduce batch size, switch to lower precision, or use a GPU with more memory"
        elif prob > 0.7:
            level = "medium"
            rec = "Consider reducing batch size or enabling memory-saving optimizations"
        else:
            level = "low"
            rec = "OOM risk is acceptable"

        return OOMRiskPrediction(
            job_id=job.get("job_id", ""),
            gpu_index=job_gpu,
            current_memory_used_gb=round(avail_mem * (1 - prob) if avail_mem > 0 else 0, 1),
            peak_memory_predicted_gb=round(peak_mem, 1),
            available_memory_gb=round(avail_mem, 1),
            oom_probability=round(min(prob, 1.0), 3),
            risk_level=level,
            recommendation=rec,
        )


class ThermalRiskPredictor:
    def predict(self, gpu_snapshot: dict) -> list[ThermalRiskPrediction]:
        results = []
        devices = gpu_snapshot.get("devices", [])
        for d in devices:
            temp = d.get("temperature_celsius", 50)
            util = d.get("utilization_percent", 50)
            power = d.get("power_draw_watts", 200)

            temp_trend = temp + util * 0.05 + power * 0.01
            peak = temp + (temp_trend - temp) * 2
            if peak > 85:
                throttle_prob = (peak - 85) / 15
            else:
                throttle_prob = 0.0

            if throttle_prob > 0:
                time_to = (85 - temp) / max(temp_trend - temp, 0.01)
            else:
                time_to = 999.0

            if throttle_prob > 0.5:
                level = "high"
                rec = "Reduce power limit, increase cooling, or migrate workload"
            elif throttle_prob > 0.2:
                level = "medium"
                rec = "Monitor temperature, consider reducing clock speed"
            else:
                level = "low"
                rec = "Temperature within normal range"

            results.append(ThermalRiskPrediction(
                node=d.get("node", "unknown"),
                gpu_index=d.get("index", 0),
                current_temperature_c=round(temp, 1),
                predicted_peak_temperature_c=round(min(peak, 105), 1),
                thermal_throttle_probability=round(min(throttle_prob, 1.0), 3),
                time_to_throttle_minutes=round(time_to, 1),
                risk_level=level,
                recommendation=rec,
            ))
        return results


class DemandBurstDetector:
    def __init__(self, window_size: int = 100) -> None:
        self._window: deque = deque(maxlen=window_size)

    def detect(self, metric_history: list[float] | None = None, threshold: float = 2.0) -> DemandBurstDetection:
        if metric_history:
            self._window.extend(metric_history)
        if not self._window:
            self._window.extend([random.uniform(10, 50) for _ in range(20)])

        values = list(self._window)
        current = values[-1]
        mean = np.mean(values)
        std = np.std(values) or 1.0
        z_score = (current - mean) / std

        burst = abs(z_score) > threshold
        magnitude = min(abs(z_score) / threshold, 5.0)

        if burst:
            severity = "critical" if magnitude > 3 else "warning" if magnitude > 2 else "info"
        else:
            severity = "info"

        return DemandBurstDetection(
            burst_detected=burst,
            burst_start_time=datetime.now(timezone.utc).isoformat() if burst else "",
            burst_magnitude=round(magnitude, 2),
            burst_duration_seconds=round(random.uniform(30, 600), 0) if burst else 0.0,
            affected_metrics=["submission_rate", "queue_depth"] if burst else [],
            trigger_threshold=threshold,
            severity=severity,
        )

    def update(self, metric_value: float) -> None:
        self._window.append(metric_value)


class ActionImpactForecaster:
    def forecast(self, action_type: str, target_state: dict | None = None) -> ActionImpactForecast:
        impacts = {
            "placement": (5.0, 0.5, 0.0, 2.0, 0.1),
            "scale_up": (10.0, 0.0, -5.0, 5.0, 0.05),
            "scale_down": (-15.0, 10.0, 10.0, -10.0, 0.1),
            "preempt": (20.0, 8.0, 5.0, -5.0, 0.3),
            "consolidate": (15.0, 15.0, 12.0, -3.0, 0.15),
            "right_size": (8.0, 5.0, 8.0, 0.0, 0.08),
        }
        util_change, mem_free, cost_save, perf_impact, risk = impacts.get(
            action_type, (0.0, 0.0, 0.0, 0.0, 0.1)
        )

        noise = random.gauss(0, abs(util_change) * 0.1)
        confidence = max(0.3, min(0.95, 0.8 - abs(noise) / max(util_change, 1) * 0.3))

        return ActionImpactForecast(
            action_type=action_type,
            description=f"{action_type.replace('_', ' ').title()} action",
            expected_gpu_utilization_change=round(util_change + noise, 1),
            expected_memory_freed_gb=round(mem_free + random.gauss(0, 1), 1),
            expected_cost_savings=round(cost_save + random.gauss(0, 0.5), 2),
            expected_performance_impact=round(perf_impact + random.gauss(0, 1), 1),
            risk_of_disruption=round(risk, 2),
            confidence=round(confidence, 2),
            recommended=risk < 0.3,
        )


class ComprehensivePredictionService:
    def __init__(self) -> None:
        self._queue_predictor = QueuePressurePredictor()
        self._jct_predictor = JCTPredictor()
        self._oom_predictor = OOMRiskPredictor()
        self._thermal_predictor = ThermalRiskPredictor()
        self._burst_detector = DemandBurstDetector()
        self._impact_forecaster = ActionImpactForecaster()

    def predict_all(self, cluster_id: str = "") -> ComprehensivePrediction:
        try:
            monitor = GPUMonitor()
            gpu_snap = monitor.collect()
            gpu_dict = {
                "total_gpus": gpu_snap.total_gpus,
                "devices": [
                    {
                        "index": d.index,
                        "memory_total_mb": d.memory_total_mb,
                        "memory_used_mb": d.memory_used_mb,
                        "utilization_percent": d.utilization_gpu_percent,
                        "temperature_celsius": d.temperature_celsius,
                        "power_draw_watts": d.power_draw_watts,
                    }
                    for d in gpu_snap.devices
                ],
            }
        except Exception as exc:
            logger.debug("GPU collection failed: %s", exc)
            gpu_dict = {"total_gpus": 0, "devices": []}

        queue_fc = self._queue_predictor.predict()
        jcts = [
            self._jct_predictor.predict({
                "job_id": f"job-{i}", "gpu_required": random.randint(1, 4),
                "memory_required_gb": random.uniform(4, 64),
                "framework": random.choice(["pytorch", "tensorflow", "jax"]),
                "dataset_size_gb": random.uniform(1, 500),
                "model_size_gb": random.uniform(0.5, 30),
            })
            for i in range(random.randint(2, 5))
        ]
        ooms = [
            self._oom_predictor.predict({
                "job_id": f"job-{i}", "gpu_index": random.randint(0, max(0, gpu_dict["total_gpus"] - 1)),
                "memory_required_gb": random.uniform(4, 80),
                "model_size_gb": random.uniform(0.5, 20),
            }, gpu_dict)
            for i in range(random.randint(1, 4))
        ]
        thermals = self._thermal_predictor.predict(gpu_dict)
        bursts = [self._burst_detector.detect()]
        impacts = [self._impact_forecaster.forecast(at) for at in
                   ["placement", "scale_up", "consolidate", "right_size"]]

        risk_scores = []
        if queue_fc.congestion_probability > 0.5:
            risk_scores.append(0.3 * queue_fc.congestion_probability)
        for o in ooms:
            risk_scores.append(0.4 * o.oom_probability)
        for t in thermals:
            risk_scores.append(0.3 * t.thermal_throttle_probability)
        for b in bursts:
            if b.burst_detected:
                risk_scores.append(0.2 * b.burst_magnitude)
        overall = min(1.0, sum(risk_scores)) if risk_scores else 0.0

        n_risks = sum(1 for o in ooms if o.risk_level == "high") + sum(1 for t in thermals if t.risk_level == "high")
        summary = f"Overall risk: {overall:.0%}. "
        if n_risks > 0:
            summary += f"{n_risks} high-risk predictions. "
        if queue_fc.pressure_level in ("high", "critical"):
            summary += "Queue pressure elevated. "
        summary += "Recommend proactive monitoring."

        return ComprehensivePrediction(
            cluster_id=cluster_id,
            queue_forecast=queue_fc,
            jct_predictions=jcts,
            oom_risks=ooms,
            thermal_risks=thermals,
            demand_bursts=bursts,
            action_impacts=impacts,
            overall_risk_score=round(overall, 3),
            summary=summary,
        )


from datetime import datetime, timezone
