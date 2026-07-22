from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GpuState:
    index: int
    model: str = "NVIDIA H100-SXM-80GB"
    memory_total_gib: float = 80.0
    memory_used_gib: float = 0.0
    engine_util_pct: float = 0.0
    tensor_activity_pct: float = 0.0
    dram_activity_pct: float = 0.0
    power_draw_watts: float = 30.0
    power_limit_watts: float = 400.0
    gpu_temp_celsius: float = 35.0
    memory_temp_celsius: float = 30.0
    fan_speed_pct: float = 30.0
    pcie_tx_gbps: float = 0.0
    pcie_rx_gbps: float = 0.0
    nvlink_tx_gbps: float = 0.0
    nvlink_rx_gbps: float = 0.0
    clock_mhz: float = 1000.0
    mem_clock_mhz: float = 1500.0
    voltage_mv: float = 700.0
    xid_errors: int = 0
    ecc_errors: int = 0
    retired_pages: int = 0
    pcie_replay_count: int = 0


@dataclass
class WorkloadProfile:
    name: str
    gpu_util_target: float
    memory_gib_target: float
    duration_seconds: float
    power_draw_watts: float
    tensor_core_intensity: float
    memory_intensity: float
    batch_size: int = 1
    concurrency: int = 1


WORKLOAD_PROFILES: dict[str, WorkloadProfile] = {
    "llm_inference": WorkloadProfile(
        name="LLM Inference", gpu_util_target=45.0, memory_gib_target=32.0,
        duration_seconds=3600, power_draw_watts=250, tensor_core_intensity=0.7,
        memory_intensity=0.5, batch_size=1, concurrency=8,
    ),
    "llm_training": WorkloadProfile(
        name="LLM Training", gpu_util_target=92.0, memory_gib_target=72.0,
        duration_seconds=86400, power_draw_watts=380, tensor_core_intensity=0.95,
        memory_intensity=0.8, batch_size=64, concurrency=1,
    ),
    "cnn_training": WorkloadProfile(
        name="CNN Training", gpu_util_target=88.0, memory_gib_target=40.0,
        duration_seconds=43200, power_draw_watts=320, tensor_core_intensity=0.85,
        memory_intensity=0.4, batch_size=128, concurrency=1,
    ),
    "batch_inference": WorkloadProfile(
        name="Batch Inference", gpu_util_target=75.0, memory_gib_target=24.0,
        duration_seconds=7200, power_draw_watts=200, tensor_core_intensity=0.5,
        memory_intensity=0.3, batch_size=32, concurrency=4,
    ),
    "data_processing": WorkloadProfile(
        name="Data Processing", gpu_util_target=35.0, memory_gib_target=16.0,
        duration_seconds=1800, power_draw_watts=150, tensor_core_intensity=0.2,
        memory_intensity=0.7, batch_size=1, concurrency=2,
    ),
    "hpo_search": WorkloadProfile(
        name="HPO Search", gpu_util_target=65.0, memory_gib_target=48.0,
        duration_seconds=14400, power_draw_watts=300, tensor_core_intensity=0.6,
        memory_intensity=0.6, batch_size=16, concurrency=4,
    ),
}


@dataclass
class ThermalNode:
    ambient_temp: float = 25.0
    thermal_resistance_kw: float = 0.15
    thermal_capacity_jk: float = 500.0
    fan_efficiency: float = 0.8
    max_safe_temp: float = 88.0
    throttling_temp: float = 85.0

    def compute_steady_state_temp(self, power_watts: float, fan_speed: float) -> float:
        delta_t = power_watts * self.thermal_resistance_kw * (1.0 - fan_speed / 100 * self.fan_efficiency * 0.3)
        return self.ambient_temp + delta_t

    def compute_transient_temp(self, current_temp: float, power_watts: float, fan_speed: float, dt: float = 1.0) -> float:
        steady = self.compute_steady_state_temp(power_watts, fan_speed)
        tau = self.thermal_capacity_jk * self.thermal_resistance_kw
        return current_temp + (steady - current_temp) * (1.0 - math.exp(-dt / max(tau, 1)))


class GpuSimulationEngine:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.thermal = ThermalNode()
        self.time: float = 0.0

    def create_gpu(self, index: int, model: str = "NVIDIA H100-SXM-80GB") -> GpuState:
        mem_map = {"NVIDIA H100-SXM-80GB": 80.0, "NVIDIA A100-SXM-80GB": 80.0, "NVIDIA A100-SXM-40GB": 40.0, "NVIDIA RTX 4090": 24.0}
        power_map = {"NVIDIA H100-SXM-80GB": 400.0, "NVIDIA A100-SXM-80GB": 400.0, "NVIDIA A100-SXM-40GB": 250.0, "NVIDIA RTX 4090": 350.0}
        return GpuState(
            index=index, model=model,
            memory_total_gib=mem_map.get(model, 80.0),
            power_limit_watts=power_map.get(model, 400.0),
        )

    def simulate_workload(
        self,
        gpu: GpuState,
        profile: WorkloadProfile,
        workload_progress: float = 0.0,
        noise_scale: float = 0.05,
    ) -> GpuState:
        phase = math.sin(workload_progress * math.pi * 2) * 0.5 + 0.5
        noise = float(self.rng.normal(0, noise_scale))

        base_util = profile.gpu_util_target * (0.8 + 0.4 * phase)
        gpu.engine_util_pct = max(0, min(100, base_util + float(self.rng.normal(0, 5))))
        gpu.tensor_activity_pct = max(0, min(100, gpu.engine_util_pct * profile.tensor_core_intensity + float(self.rng.normal(0, 5))))
        gpu.dram_activity_pct = max(0, min(100, gpu.engine_util_pct * profile.memory_intensity + float(self.rng.normal(0, 5))))

        mem_base = profile.memory_gib_target * (0.9 + 0.2 * phase)
        gpu.memory_used_gib = max(0, min(gpu.memory_total_gib, mem_base + float(self.rng.normal(0, 2))))

        util_factor = gpu.engine_util_pct / 100.0
        base_power = profile.power_draw_watts * (0.7 + 0.6 * util_factor)
        gpu.power_draw_watts = max(0, min(gpu.power_limit_watts, base_power + float(self.rng.normal(0, 15))))

        gpu.gpu_temp_celsius = self.thermal.compute_transient_temp(
            gpu.gpu_temp_celsius, gpu.power_draw_watts, gpu.fan_speed_pct, dt=5.0
        )
        gpu.memory_temp_celsius = gpu.gpu_temp_celsius + float(self.rng.uniform(-3, 5))

        target_fan = 30.0 + 60.0 * max(0, (gpu.gpu_temp_celsius - 40) / 50)
        gpu.fan_speed_pct = max(10, min(100, target_fan + float(self.rng.normal(0, 3))))

        clock_boost = min(1.0, (70.0 - max(0, gpu.gpu_temp_celsius - 40)) / 50)
        base_clock = 1200.0 if "H100" in gpu.model else 1000.0
        gpu.clock_mhz = base_clock * (0.6 + 0.4 * util_factor) * (0.7 + 0.3 * clock_boost)
        gpu.mem_clock_mhz = 1500.0 * (0.8 + 0.2 * util_factor)

        gpu.voltage_mv = 700.0 + 200.0 * util_factor

        throughput_gbps = 50.0 * util_factor * profile.concurrency
        gpu.pcie_tx_gbps = throughput_gbps * 0.4 + float(self.rng.normal(0, 1))
        gpu.pcie_rx_gbps = throughput_gbps * 0.6 + float(self.rng.normal(0, 1))
        gpu.nvlink_tx_gbps = throughput_gbps * 0.8 * profile.tensor_core_intensity + float(self.rng.normal(0, 2))
        gpu.nvlink_rx_gbps = throughput_gbps * 0.8 * profile.tensor_core_intensity + float(self.rng.normal(0, 2))

        stress = gpu.gpu_temp_celsius / 85.0
        if self.rng.random() < 0.001 * stress:
            gpu.xid_errors += 1
        if self.rng.random() < 0.002 * stress:
            gpu.ecc_errors += int(self.rng.poisson(1))
        if self.rng.random() < 0.0005 * stress:
            gpu.retired_pages += 1
        if self.rng.random() < 0.003 * stress:
            gpu.pcie_replay_count += 1

        return gpu

    def simulate_node(
        self,
        num_gpus: int = 8,
        gpu_model: str = "NVIDIA H100-SXM-80GB",
        workload_type: str = "llm_inference",
        duration_steps: int = 60,
        step_seconds: float = 5.0,
    ) -> list[list[GpuState]]:
        profile = WORKLOAD_PROFILES.get(workload_type, WORKLOAD_PROFILES["llm_inference"])
        gpus = [self.create_gpu(i, gpu_model) for i in range(num_gpus)]
        history: list[list[GpuState]] = []

        for step in range(duration_steps):
            progress = step / max(duration_steps, 1)
            step_gpus: list[GpuState] = []
            for gpu in gpus:
                sim_gpu = GpuState(
                    index=gpu.index, model=gpu.model,
                    memory_total_gib=gpu.memory_total_gib,
                    power_limit_watts=gpu.power_limit_watts,
                    fan_speed_pct=gpu.fan_speed_pct,
                    gpu_temp_celsius=gpu.gpu_temp_celsius,
                    xid_errors=gpu.xid_errors,
                    ecc_errors=gpu.ecc_errors,
                    retired_pages=gpu.retired_pages,
                    pcie_replay_count=gpu.pcie_replay_count,
                )
                sim_gpu = self.simulate_workload(sim_gpu, profile, progress)
                step_gpus.append(sim_gpu)
            history.append(step_gpus)

        return history

    def simulate_failure_scenario(
        self,
        num_gpus: int = 8,
        scenario: str = "thermal_runaway",
    ) -> dict[str, Any]:
        gpus = [self.create_gpu(i) for i in range(num_gpus)]
        profile = WORKLOAD_PROFILES["llm_training"]
        timeline: list[dict[str, Any]] = []
        failure_step = -1
        root_cause = ""

        for step in range(200):
            progress = step / 200
            for gpu in gpus:
                self.simulate_workload(gpu, profile, progress)

                if scenario == "thermal_runaway" and step > 50:
                    if gpu.index == 0:
                        gpu.power_draw_watts *= 1.02
                        gpu.gpu_temp_celsius = self.thermal.compute_transient_temp(
                            gpu.gpu_temp_celsius, gpu.power_draw_watts, gpu.fan_speed_pct, dt=5.0
                        )

                elif scenario == "memory_leak" and step > 30:
                    if gpu.index == 0:
                        leak = profile.memory_gib_target * 0.005
                        gpu.memory_used_gib = min(gpu.memory_total_gib * 1.05, gpu.memory_used_gib + leak)

                elif scenario == "xid_storm" and step > 20:
                    if self.rng.random() < 0.01 * (1 + step / 100):
                        gpu.xid_errors += int(self.rng.poisson(2))
                        gpu.ecc_errors += int(self.rng.poisson(3))

                elif scenario == "power_spike":
                    if step == 80 and gpu.index == 0:
                        gpu.power_draw_watts = gpu.power_limit_watts * 1.3
                        gpu.gpu_temp_celsius = min(105, gpu.gpu_temp_celsius + 20)

            avg_temp = sum(g.gpu_temp_celsius for g in gpus) / len(gpus)
            max_temp = max(g.gpu_temp_celsius for g in gpus)
            total_power = sum(g.power_draw_watts for g in gpus)
            total_xid = sum(g.xid_errors for g in gpus)
            total_ecc = sum(g.ecc_errors for g in gpus)
            max_mem_util = max(g.memory_used_gib / g.memory_total_gib * 100 for g in gpus)

            if step == 60:
                timeline.append({
                    "step": step, "phase": "normal", "avg_temp": round(avg_temp, 1),
                    "max_temp": round(max_temp, 1), "total_power": round(total_power, 1),
                })
            elif step == 100:
                status = "degrading" if scenario != "normal" else "normal"
                timeline.append({
                    "step": step, "phase": status, "avg_temp": round(avg_temp, 1),
                    "max_temp": round(max_temp, 1), "total_power": round(total_power, 1),
                    "total_xid": total_xid, "total_ecc": total_ecc,
                    "max_mem_util": round(max_mem_util, 1),
                })

            if max_temp > 88 or max_mem_util > 98 or total_xid > 20:
                failure_step = step
                root_cause = (
                    "Thermal runaway" if max_temp > 88 else
                    "Memory exhaustion" if max_mem_util > 98 else
                    "XID error storm"
                )
                timeline.append({
                    "step": step, "phase": "failure", "root_cause": root_cause,
                    "avg_temp": round(avg_temp, 1), "max_temp": round(max_temp, 1),
                    "total_power": round(total_power, 1), "total_xid": total_xid,
                    "total_ecc": total_ecc, "max_mem_util": round(max_mem_util, 1),
                })
                for gpu in gpus:
                    gpu.engine_util_pct = 0
                    gpu.power_draw_watts = gpu.power_limit_watts * 0.1
                break

        return {
            "scenario": scenario,
            "workload": profile.name,
            "num_gpus": num_gpus,
            "failure_detected": failure_step >= 0,
            "failure_step": failure_step,
            "root_cause": root_cause,
            "time_to_failure_seconds": failure_step * 5 if failure_step >= 0 else None,
            "timeline": timeline,
            "gpu_model": gpus[0].model,
            "simulation_duration_steps": step + 1,
            "final_state": {
                "avg_temp": round(sum(g.gpu_temp_celsius for g in gpus) / len(gpus), 1),
                "max_temp": round(max(g.gpu_temp_celsius for g in gpus), 1),
                "avg_util": round(sum(g.engine_util_pct for g in gpus) / len(gpus), 1),
                "total_power": round(sum(g.power_draw_watts for g in gpus), 1),
                "total_xid": sum(g.xid_errors for g in gpus),
                "total_ecc": sum(g.ecc_errors for g in gpus),
            },
        }


class DigitalTwinSimulationService:
    def __init__(self) -> None:
        self.engine = GpuSimulationEngine()

    def simulate(self, num_gpus: int = 8, gpu_model: str = "NVIDIA H100-SXM-80GB",
                 workload_type: str = "llm_inference", duration_steps: int = 60) -> dict:
        history = self.engine.simulate_node(num_gpus, gpu_model, workload_type, duration_steps)
        latest = history[-1] if history else []
        return {
            "simulation_id": uuid.uuid4().hex[:12],
            "workload": WORKLOAD_PROFILES.get(workload_type, WORKLOAD_PROFILES["llm_inference"]).name,
            "num_gpus": num_gpus,
            "gpu_model": gpu_model,
            "steps": duration_steps,
            "current_state": [
                {
                    "gpu_index": g.index,
                    "engine_util_pct": round(g.engine_util_pct, 1),
                    "memory_used_gib": round(g.memory_used_gib, 1),
                    "memory_pct": round(g.memory_used_gib / g.memory_total_gib * 100, 1),
                    "gpu_temp": round(g.gpu_temp_celsius, 1),
                    "power_watts": round(g.power_draw_watts, 1),
                    "tensor_activity_pct": round(g.tensor_activity_pct, 1),
                    "dram_activity_pct": round(g.dram_activity_pct, 1),
                    "clock_mhz": round(g.clock_mhz, 0),
                    "fan_speed_pct": round(g.fan_speed_pct, 1),
                    "xid_errors": g.xid_errors,
                    "ecc_errors": g.ecc_errors,
                }
                for g in latest
            ],
            "aggregate": {
                "avg_util": round(sum(g.engine_util_pct for g in latest) / max(len(latest), 1), 1),
                "avg_temp": round(sum(g.gpu_temp_celsius for g in latest) / max(len(latest), 1), 1),
                "max_temp": round(max(g.gpu_temp_celsius for g in latest), 1),
                "total_power": round(sum(g.power_draw_watts for g in latest), 1),
                "total_memory_used_gib": round(sum(g.memory_used_gib for g in latest), 1),
                "total_xid_errors": sum(g.xid_errors for g in latest),
                "total_ecc_errors": sum(g.ecc_errors for g in latest),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def simulate_failure(self, scenario: str = "thermal_runaway", num_gpus: int = 8) -> dict:
        valid = ["thermal_runaway", "memory_leak", "xid_storm", "power_spike"]
        if scenario not in valid:
            scenario = self.engine.rng.choice(valid)
        return self.engine.simulate_failure_scenario(num_gpus, scenario)

    def list_profiles(self) -> list[dict]:
        return [
            {
                "name": k,
                "label": v.name,
                "gpu_util_target": v.gpu_util_target,
                "memory_gib_target": v.memory_gib_target,
                "power_draw_watts": v.power_draw_watts,
                "duration_seconds": v.duration_seconds,
                "tensor_core_intensity": v.tensor_core_intensity,
                "memory_intensity": v.memory_intensity,
            }
            for k, v in WORKLOAD_PROFILES.items()
        ]

    def health(self) -> dict:
        return {
            "status": "healthy",
            "workload_profiles": len(WORKLOAD_PROFILES),
            "failure_scenarios": ["thermal_runaway", "memory_leak", "xid_storm", "power_spike"],
        }
