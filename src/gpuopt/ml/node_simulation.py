from __future__ import annotations

import enum
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CoolingType(enum.Enum):
    AIR = "air"
    DIRECT_LIQUID = "direct_liquid"
    IMMERSION = "immersion"


class TopologyType(enum.Enum):
    SINGLE_NODE = "single_node"
    DGX_H100 = "dgx_h100"
    DGX_A100 = "dgx_a100"
    CUSTOM = "custom"


@dataclass
class GpuSpec:
    model: str = "NVIDIA H100-SXM-80GB"
    memory_gib: float = 80.0
    tdp_watts: float = 400.0
    base_clock_mhz: float = 1830.0
    boost_clock_mhz: float = 1980.0
    mem_clock_mhz: float = 1593.0
    nvlink_bandwidth_gbps: float = 900.0
    pcie_bandwidth_gbps: float = 128.0
    tensor_tflops_fp16: float = 1979.0
    tensor_tflops_fp8: float = 3958.0


GPU_SPECS: dict[str, GpuSpec] = {
    "NVIDIA H100-SXM-80GB": GpuSpec(tdp_watts=400.0, memory_gib=80.0),
    "NVIDIA A100-SXM-80GB": GpuSpec(model="NVIDIA A100-SXM-80GB", tdp_watts=400.0, memory_gib=80.0, base_clock_mhz=1410, boost_clock_mhz=1410, mem_clock_mhz=1215, nvlink_bandwidth_gbps=600, pcie_bandwidth_gbps=64, tensor_tflops_fp16=624, tensor_tflops_fp8=1248),
    "NVIDIA A100-SXM-40GB": GpuSpec(model="NVIDIA A100-SXM-40GB", tdp_watts=250.0, memory_gib=40.0, base_clock_mhz=1410, boost_clock_mhz=1410, mem_clock_mhz=1215, nvlink_bandwidth_gbps=600, pcie_bandwidth_gbps=64, tensor_tflops_fp16=624),
    "NVIDIA RTX 4090": GpuSpec(model="NVIDIA RTX 4090", tdp_watts=350.0, memory_gib=24.0, base_clock_mhz=2235, boost_clock_mhz=2520, mem_clock_mhz=1313, nvlink_bandwidth_gbps=0, pcie_bandwidth_gbps=128, tensor_tflops_fp16=330),
    "NVIDIA RTX 6000 Ada": GpuSpec(model="NVIDIA RTX 6000 Ada", tdp_watts=300.0, memory_gib=48.0, base_clock_mhz=915, boost_clock_mhz=2535, mem_clock_mhz=1250, nvlink_bandwidth_gbps=0, pcie_bandwidth_gbps=128, tensor_tflops_fp16=364),
    "AMD MI300X": GpuSpec(model="AMD MI300X", tdp_watts=750.0, memory_gib=192.0, base_clock_mhz=1700, boost_clock_mhz=2100, mem_clock_mhz=1200, nvlink_bandwidth_gbps=896, pcie_bandwidth_gbps=128, tensor_tflops_fp16=1307),
}


@dataclass
class CoolantLoop:
    flow_rate_lpm: float = 20.0
    inlet_temp_c: float = 25.0
    outlet_temp_c: float = 45.0
    pump_power_watts: float = 200.0
    heat_exchanger_capacity_kw: float = 60.0


@dataclass
class CoolingSystem:
    cooling_type: CoolingType = CoolingType.AIR
    ambient_temp_c: float = 22.0
    airflow_cfm: float = 500.0
    liquid_loop: CoolantLoop | None = None
    chiller_power_kw: float = 0.0
    pue: float = 1.3

    @property
    def effective_ambient(self) -> float:
        if self.cooling_type == CoolingType.IMMERSION:
            return self.ambient_temp_c + 5.0
        elif self.cooling_type == CoolingType.DIRECT_LIQUID:
            return self.ambient_temp_c + 8.0
        return self.ambient_temp_c + 10.0


@dataclass
class PowerDistributionUnit:
    max_power_kw: float = 30.0
    efficiency: float = 0.96
    circuits: int = 3
    circuit_capacity_kw: float = 10.0
    circuit_load_kw: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def total_load_kw(self) -> float:
        return sum(self.circuit_load_kw)

    @property
    def headroom_kw(self) -> float:
        return self.max_power_kw - self.total_load_kw

    def can_serve(self, additional_kw: float) -> bool:
        if additional_kw + self.total_load_kw > self.max_power_kw:
            return False
        for i in range(self.circuits):
            if self.circuit_load_kw[i] + additional_kw / self.circuits <= self.circuit_capacity_kw:
                return True
        return False

    def assign_load(self, kw: float) -> int:
        target_circuit = min(
            range(self.circuits), key=lambda i: self.circuit_load_kw[i],
        )
        self.circuit_load_kw[target_circuit] += kw / self.circuits
        return target_circuit


@dataclass
class GpuDegradation:
    total_hours_operational: float = 0.0
    total_power_cycles: int = 0
    thermal_cycles_above_80c: int = 0
    peak_temp_recorded: float = 35.0
    accumulated_thermal_stress: float = 0.0
    pcie_replay_count: int = 0
    memory_remapped_rows: int = 0
    voltage_droop_events: int = 0

    @property
    def wear_factor(self) -> float:
        base = 0.02 * (self.total_hours_operational / 50000.0)
        thermal = 0.01 * (self.thermal_cycles_above_80c / 1000.0)
        stress = self.accumulated_thermal_stress / 1000.0
        cycles = 0.005 * (self.total_power_cycles / 5000.0)
        return min(1.0, base + thermal + stress + cycles)

    @property
    def mtbf_multiplier(self) -> float:
        return 1.0 - self.wear_factor


@dataclass
class SimGpu:
    index: int
    node_id: str
    spec: GpuSpec
    memory_used_gib: float = 0.0
    engine_util_pct: float = 0.0
    tensor_activity_pct: float = 0.0
    dram_activity_pct: float = 0.0
    power_draw_watts: float = 30.0
    power_cap_watts: float = 400.0
    gpu_temp_c: float = 35.0
    memory_temp_c: float = 30.0
    fan_speed_pct: float = 30.0
    clock_mhz: float = 1000.0
    mem_clock_mhz: float = 1500.0
    voltage_mv: float = 700.0
    xid_errors: int = 0
    ecc_corrected: int = 0
    ecc_uncorrected: int = 0
    retired_pages: int = 0
    nvlink_tx_gbps: float = 0.0
    nvlink_rx_gbps: float = 0.0
    pcie_tx_gbps: float = 0.0
    pcie_rx_gbps: float = 0.0
    is_faulted: bool = False
    fault_reason: str = ""
    degradation: GpuDegradation = field(default_factory=GpuDegradation)

    @property
    def memory_pct(self) -> float:
        return (self.memory_used_gib / self.spec.memory_gib) * 100 if self.spec.memory_gib > 0 else 0.0


@dataclass
class SimNode:
    node_id: str
    gpus: list[SimGpu]
    cpu_cores: int = 64
    cpu_util_pct: float = 0.0
    cpu_vendor: str = ""
    system_memory_gib: float = 512.0
    system_memory_used_gib: float = 0.0
    memory_type: str = ""
    memory_bandwidth_gbps: float = 0.0
    ecc_capable: bool = False
    cooling: CoolingSystem = field(default_factory=CoolingSystem)
    nic_bandwidth_gbps: float = 400.0
    nic_util_pct: float = 0.0
    is_on: bool = True
    failure_time_hours: float = float("inf")

    @property
    def total_gpu_power_watts(self) -> float:
        return sum(g.power_draw_watts for g in self.gpus)

    @property
    def total_power_watts(self) -> float:
        gpu_power = self.total_gpu_power_watts
        cpu_power = 150.0 * (self.cpu_util_pct / 100.0) + 50.0
        mem_power = 10.0 * (self.system_memory_used_gib / self.system_memory_gib)
        nic_power = 25.0 * (self.nic_util_pct / 100.0)
        cooling_overhead = gpu_power * (self.cooling.pue - 1.0)
        return gpu_power + cpu_power + mem_power + nic_power + cooling_overhead


@dataclass
class ClusterTopology:
    topology_type: TopologyType = TopologyType.CUSTOM
    nodes: list[SimNode] = field(default_factory=list)
    pdu: PowerDistributionUnit = field(default_factory=PowerDistributionUnit)
    interconnect_type: str = "nvlink"
    total_nic_bandwidth_gbps: float = 0.0

    @property
    def total_gpus(self) -> int:
        return sum(len(n.gpus) for n in self.nodes)

    @property
    def total_gpu_power_watts(self) -> float:
        return sum(n.total_gpu_power_watts for n in self.nodes)

    @property
    def total_cluster_power_watts(self) -> float:
        return sum(n.total_power_watts for n in self.nodes)

    def get_gpu(self, node_idx: int, gpu_idx: int) -> SimGpu:
        return self.nodes[node_idx].gpus[gpu_idx]

    def build_dgx_h100(self, num_nodes: int = 1) -> ClusterTopology:
        self.topology_type = TopologyType.DGX_H100
        self.interconnect_type = "nvlink"
        self.pdu = PowerDistributionUnit(
            max_power_kw=num_nodes * 12.0, circuits=max(3, num_nodes),
        )
        for n in range(num_nodes):
            node_id = f"dgx-h100-{n:02d}"
            cooling = CoolingSystem(cooling_type=CoolingType.DIRECT_LIQUID, pue=1.15)
            gpus = []
            for g in range(8):
                gpu = SimGpu(
                    index=g, node_id=node_id, spec=GPU_SPECS["NVIDIA H100-SXM-80GB"],
                    power_cap_watts=400.0,
                )
                gpus.append(gpu)
            node = SimNode(node_id=node_id, gpus=gpus, cooling=cooling)
            self.nodes.append(node)
            self.pdu.assign_load(node.total_power_watts / 1000.0)
            self.total_nic_bandwidth_gbps += 400.0
        return self

    def build_rtx_cluster(self, num_gpus: int = 8) -> ClusterTopology:
        self.topology_type = TopologyType.CUSTOM
        self.interconnect_type = "pcie"
        cooling = CoolingSystem(cooling_type=CoolingType.AIR, pue=1.3)
        gpus = [
            SimGpu(index=i, node_id="rtx-node-00", spec=GPU_SPECS["NVIDIA RTX 4090"],
                   power_cap_watts=350.0)
            for i in range(num_gpus)
        ]
        node = SimNode(node_id="rtx-node-00", gpus=gpus, cooling=cooling,
                       cpu_cores=16, system_memory_gib=128.0)
        self.nodes = [node]
        self.pdu = PowerDistributionUnit(max_power_kw=5.0)
        self.total_nic_bandwidth_gbps = 100.0
        return self


class EnhancedSimulationEngine:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.topology: ClusterTopology | None = None
        self.time_hours: float = 0.0
        self.time_step_seconds: float = 5.0
        self.step_count: int = 0
        self._event_log: list[dict] = []

    def reset_event_log(self) -> None:
        self._event_log = []

    @property
    def event_log(self) -> list[dict]:
        return self._event_log

    def log_event(self, event_type: str, node_id: str, gpu_idx: int, message: str, **kwargs) -> None:
        event: dict = {
            "step": self.step_count,
            "time_hours": round(self.time_hours, 4),
            "event_type": event_type,
            "node_id": node_id,
            "gpu_index": gpu_idx,
            "message": message,
        }
        event.update(kwargs)
        self._event_log.append(event)
        logger.debug("Event: %s on %s GPU%d: %s", event_type, node_id, gpu_idx, message)

    def init_topology(self, topology: ClusterTopology) -> None:
        self.topology = topology
        self.reset_event_log()

    def compute_node_thermal(
        self, node: SimNode, gpu: SimGpu, dt: float = 5.0,
    ) -> tuple[float, float]:
        ambient = node.cooling.effective_ambient
        power = gpu.power_draw_watts
        pue = node.cooling.pue

        if node.cooling.cooling_type == CoolingType.DIRECT_LIQUID:
            thermal_resistance = 0.08
            fan_effectiveness = 0.95
        elif node.cooling.cooling_type == CoolingType.IMMERSION:
            thermal_resistance = 0.05
            fan_effectiveness = 1.0
        else:
            thermal_resistance = 0.15
            fan_effectiveness = 0.7

        airflow_factor = gpu.fan_speed_pct / 100.0 * fan_effectiveness
        delta_t_steady = power * thermal_resistance * (1.0 - airflow_factor * 0.4)
        steady_temp = ambient + delta_t_steady

        thermal_capacity = 500.0 * (1.0 + 0.5 * pue)
        tau = thermal_capacity * thermal_resistance
        new_temp = gpu.gpu_temp_c + (steady_temp - gpu.gpu_temp_c) * (1.0 - math.exp(-dt / max(tau, 1)))

        mem_temp = new_temp + self.rng.uniform(-2, 4)
        return new_temp, mem_temp

    def compute_power(
        self,
        gpu: SimGpu,
        util_pct: float,
        tensor_activity: float,
        mem_activity: float,
    ) -> tuple[float, float, float]:
        spec = gpu.spec
        util_ratio = max(0.0, min(1.0, util_pct / 100.0))
        tensor_ratio = max(0.0, min(1.0, tensor_activity / 100.0))
        mem_ratio = max(0.0, min(1.0, mem_activity / 100.0))

        compute_power = spec.tdp_watts * 0.6 * util_ratio
        tensor_power = spec.tdp_watts * 0.25 * tensor_ratio
        mem_power = spec.tdp_watts * 0.1 * mem_ratio
        static_power = spec.tdp_watts * 0.05
        raw_power = compute_power + tensor_power + mem_power + static_power

        temp_penalty = 1.0 + max(0, gpu.gpu_temp_c - 75) * 0.005
        power_capped = min(raw_power * temp_penalty, gpu.power_cap_watts)

        temp_factor = gpu.gpu_temp_c / 85.0
        clock_ratio = gpu.clock_mhz / spec.boost_clock_mhz
        voltage = 700.0 + 200.0 * util_ratio * clock_ratio + 50.0 * temp_factor
        voltage = min(voltage, 1200.0)

        return power_capped, voltage, clock_ratio

    def compute_clock_speed(
        self, gpu: SimGpu, power_ratio: float, util_ratio: float,
    ) -> float:
        spec = gpu.spec
        temp_throttle = max(0.0, (gpu.gpu_temp_c - 85.0) / 15.0)
        power_throttle = max(0.0, (power_ratio - 0.95) * 5.0)
        throttle = min(1.0, temp_throttle + power_throttle)

        base = spec.boost_clock_mhz if util_ratio > 0.1 else spec.base_clock_mhz
        return base * (0.5 + 0.5 * util_ratio) * (1.0 - 0.3 * throttle)

    def inject_errors(
        self, gpu: SimGpu, stress: float, profile_intensity: float,
    ) -> None:
        degradation = gpu.degradation
        wear = degradation.wear_factor
        effective_stress = stress * (1.0 + wear * 2.0)

        xid_rate = 0.0005 * effective_stress * profile_intensity
        ecc_rate = 0.001 * effective_stress * profile_intensity
        retire_rate = 0.0002 * effective_stress * (1 + wear * 3)
        pcie_rate = 0.0015 * effective_stress

        if self.rng.random() < xid_rate:
            gpu.xid_errors += int(self.rng.poisson(1)) + 1
            if gpu.xid_errors > 5:
                gpu.is_faulted = True
                gpu.fault_reason = "XID error threshold exceeded"
                self.log_event("gpu_fault", gpu.node_id, gpu.index,
                               f"GPU faulted: {gpu.fault_reason}",
                               xid_errors=gpu.xid_errors)

        if self.rng.random() < ecc_rate:
            n_ecc = int(self.rng.poisson(1))
            gpu.ecc_corrected += n_ecc
            if self.rng.random() < 0.05 * (1 + wear):
                gpu.ecc_uncorrected += 1

        if self.rng.random() < retire_rate:
            gpu.retired_pages += 1
            degradation.memory_remapped_rows += 1

        if self.rng.random() < pcie_rate:
            gpu.degradation.pcie_replay_count += 1

    def update_degradation(self, gpu: SimGpu, dt_hours: float) -> None:
        d = gpu.degradation
        d.total_hours_operational += dt_hours
        temp_ratio = gpu.gpu_temp_c / 85.0
        util_ratio = gpu.engine_util_pct / 100.0
        d.accumulated_thermal_stress += temp_ratio * util_ratio * dt_hours

        if gpu.gpu_temp_c > 80:
            d.thermal_cycles_above_80c += 1

        if gpu.gpu_temp_c > d.peak_temp_recorded:
            d.peak_temp_recorded = gpu.gpu_temp_c

    def step_simulation(
        self,
        workload_fn: Any | None = None,
        profile: dict[str, float] | None = None,
    ) -> ClusterTopology:
        if self.topology is None:
            raise RuntimeError("Topology not initialized. Call init_topology first.")

        dt_seconds = self.time_step_seconds
        dt_hours = dt_seconds / 3600.0

        profile = profile or {
            "gpu_util_target": 60.0, "memory_target_pct": 50.0,
            "tensor_intensity": 0.6, "mem_intensity": 0.5,
        }

        for node in self.topology.nodes:
            if not node.is_on:
                continue

            for gpu in node.gpus:
                if gpu.is_faulted:
                    gpu.power_draw_watts = gpu.spec.tdp_watts * 0.1
                    gpu.engine_util_pct = 0.0
                    gpu.clock_mhz = 200.0
                    continue

                progress = self.step_count / 100.0
                phase_osc = math.sin(progress * math.pi * 2) * 0.5 + 0.5

                util = profile["gpu_util_target"] * (0.7 + 0.6 * phase_osc)
                gpu.engine_util_pct = max(0, min(100, util + float(self.rng.normal(0, 4))))
                gpu.tensor_activity_pct = max(0, min(100, gpu.engine_util_pct * profile["tensor_intensity"] + float(self.rng.normal(0, 3))))
                gpu.dram_activity_pct = max(0, min(100, gpu.engine_util_pct * profile["mem_intensity"] + float(self.rng.normal(0, 3))))

                mem_ratio = profile["memory_target_pct"] / 100.0
                mem_target = gpu.spec.memory_gib * mem_ratio * (0.8 + 0.4 * phase_osc)
                gpu.memory_used_gib = max(0, min(gpu.spec.memory_gib, mem_target + float(self.rng.normal(0, 1.5))))

                power, voltage, power_ratio = self.compute_power(
                    gpu, gpu.engine_util_pct, gpu.tensor_activity_pct, gpu.dram_activity_pct,
                )
                gpu.power_draw_watts = power
                gpu.voltage_mv = voltage

                new_temp, mem_temp = self.compute_node_thermal(node, gpu, dt_seconds)
                gpu.gpu_temp_c = new_temp
                gpu.memory_temp_c = mem_temp

                target_fan = 20.0 + 70.0 * max(0, (gpu.gpu_temp_c - 35) / 55)
                gpu.fan_speed_pct = max(10, min(100, target_fan + float(self.rng.normal(0, 2))))

                gpu.clock_mhz = self.compute_clock_speed(gpu, power_ratio, gpu.engine_util_pct / 100.0)
                gpu.mem_clock_mhz = gpu.spec.mem_clock_mhz * (0.7 + 0.3 * gpu.engine_util_pct / 100.0)

                bw_scale = gpu.engine_util_pct / 100.0 * gpu.tensor_activity_pct / 100.0
                gpu.pcie_tx_gbps = gpu.spec.pcie_bandwidth_gbps * 0.4 * bw_scale + float(self.rng.normal(0, 0.5))
                gpu.pcie_rx_gbps = gpu.spec.pcie_bandwidth_gbps * 0.6 * bw_scale + float(self.rng.normal(0, 0.5))
                gpu.nvlink_tx_gbps = gpu.spec.nvlink_bandwidth_gbps * 0.5 * bw_scale + float(self.rng.normal(0, 1))
                gpu.nvlink_rx_gbps = gpu.spec.nvlink_bandwidth_gbps * 0.5 * bw_scale + float(self.rng.normal(0, 1))

                stress = gpu.gpu_temp_c / 85.0 + gpu.engine_util_pct / 100.0
                self.inject_errors(gpu, stress, profile["tensor_intensity"])
                self.update_degradation(gpu, dt_hours)

            util_avg = sum(g.engine_util_pct for g in node.gpus) / max(len(node.gpus), 1)
            node.cpu_util_pct = 30.0 + 40.0 * util_avg / 100.0
            node.system_memory_used_gib = node.system_memory_gib * (0.3 + 0.4 * util_avg / 100.0)

        self.step_count += 1
        self.time_hours += dt_hours
        return self.topology

    def simulate(
        self,
        topology: ClusterTopology,
        steps: int = 100,
        profile: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        self.init_topology(topology)
        history: list[dict] = []

        for _ in range(steps):
            self.step_simulation(profile=profile)
            history.append(self.capture_snapshot())

        return {
            "simulation_id": uuid.uuid4().hex[:12],
            "topology_type": self.topology.topology_type.value if self.topology else "unknown",
            "num_nodes": len(topology.nodes),
            "total_gpus": topology.total_gpus,
            "steps": steps,
            "step_seconds": self.time_step_seconds,
            "total_simulated_hours": round(self.time_hours, 4),
            "history": history,
            "final_snapshot": history[-1] if history else {},
            "events": self._event_log,
            "gpu_models": list({g.spec.model for n in topology.nodes for g in n.gpus}),
        }

    def capture_snapshot(self) -> dict[str, Any]:
        if self.topology is None:
            return {}
        snap: dict[str, Any] = {
            "step": self.step_count,
            "time_hours": round(self.time_hours, 4),
            "nodes": [],
        }
        for node in self.topology.nodes:
            node_snap: dict[str, Any] = {
                "node_id": node.node_id,
                "is_on": node.is_on,
                "total_power_watts": round(node.total_power_watts, 1),
                "gpu_count": len(node.gpus),
                "gpus": [],
            }
            for gpu in node.gpus:
                node_snap["gpus"].append({
                    "index": gpu.index,
                    "model": gpu.spec.model,
                    "engine_util_pct": round(gpu.engine_util_pct, 1),
                    "memory_pct": round(gpu.memory_pct, 1),
                    "gpu_temp_c": round(gpu.gpu_temp_c, 1),
                    "memory_temp_c": round(gpu.memory_temp_c, 1),
                    "power_draw_watts": round(gpu.power_draw_watts, 1),
                    "power_cap_watts": round(gpu.power_cap_watts, 1),
                    "clock_mhz": round(gpu.clock_mhz, 0),
                    "fan_speed_pct": round(gpu.fan_speed_pct, 1),
                    "xid_errors": gpu.xid_errors,
                    "ecc_corrected": gpu.ecc_corrected,
                    "ecc_uncorrected": gpu.ecc_uncorrected,
                    "retired_pages": gpu.retired_pages,
                    "is_faulted": gpu.is_faulted,
                    "fault_reason": gpu.fault_reason,
                    "wear_factor": round(gpu.degradation.wear_factor, 4),
                    "mtbf_multiplier": round(gpu.degradation.mtbf_multiplier, 4),
                })
            snap["nodes"].append(node_snap)
        snap["aggregate"] = {
            "total_power_watts": round(self.topology.total_cluster_power_watts, 1),
            "total_gpus": self.topology.total_gpus,
            "faulted_gpus": sum(1 for n in self.topology.nodes for g in n.gpus if g.is_faulted),
            "total_xid_errors": sum(g.xid_errors for n in self.topology.nodes for g in n.gpus),
            "total_ecc_errors": sum(g.ecc_corrected + g.ecc_uncorrected for n in self.topology.nodes for g in n.gpus),
            "pdu_load_kw": round(self.topology.pdu.total_load_kw, 2),
            "pdu_headroom_kw": round(self.topology.pdu.headroom_kw, 2),
        }
        return snap

    def simulate_failure_scenario(
        self,
        scenario: str = "thermal_runaway",
        topology: ClusterTopology | None = None,
    ) -> dict[str, Any]:
        if topology is None:
            topo = ClusterTopology().build_dgx_h100(1)
        else:
            topo = topology

        self.init_topology(topo)
        profile = {
            "gpu_util_target": 92.0, "memory_target_pct": 90.0,
            "tensor_intensity": 0.95, "mem_intensity": 0.8,
        }
        target_gpu = topo.get_gpu(0, 0)
        failure_step = -1
        root_cause = ""

        for step in range(300):
            self.step_simulation(profile=profile)

            if scenario == "thermal_runaway" and step > 50:
                target_gpu.power_draw_watts *= 1.015
                node = topo.nodes[0]
                new_temp, _ = self.compute_node_thermal(
                    node, target_gpu, dt=self.time_step_seconds * 2,
                )
                target_gpu.gpu_temp_c = new_temp

            elif scenario == "memory_leak" and step > 30:
                leak = target_gpu.spec.memory_gib * 0.006
                target_gpu.memory_used_gib = min(
                    target_gpu.spec.memory_gib * 1.05,
                    target_gpu.memory_used_gib + leak,
                )

            elif scenario == "xid_storm" and step > 20:
                if self.rng.random() < 0.02 * (1 + step / 100):
                    target_gpu.xid_errors += int(self.rng.poisson(3))
                    target_gpu.ecc_corrected += int(self.rng.poisson(2))

            elif scenario == "power_surge":
                if step == 80:
                    target_gpu.power_draw_watts = target_gpu.power_cap_watts * 1.4
                    target_gpu.gpu_temp_c = min(110, target_gpu.gpu_temp_c + 25)

            elif scenario == "fan_failure" and step > 40:
                if target_gpu.fan_speed_pct > 80 and self.rng.random() < 0.01:
                    target_gpu.fan_speed_pct = 15.0
                    self.log_event("fan_failure", target_gpu.node_id, target_gpu.index,
                                   "Fan speed dropped to 15%")
                    scenario = "thermal_runaway"

            max_temp = max(g.gpu_temp_c for n in topo.nodes for g in n.gpus)
            max_mem = max(g.memory_pct for n in topo.nodes for g in n.gpus)
            total_xid = sum(g.xid_errors for n in topo.nodes for g in n.gpus)

            if max_temp > 90 or max_mem > 99 or total_xid > 25:
                failure_step = step
                root_cause = (
                    "Thermal runaway" if max_temp > 90 else
                    "Memory exhaustion" if max_mem > 99 else
                    "XID error storm"
                )
                target_gpu.is_faulted = True
                target_gpu.fault_reason = root_cause
                self.log_event("cluster_failure", target_gpu.node_id, target_gpu.index,
                               f"Failure: {root_cause} at step {step}")
                break

        return {
            "scenario": scenario,
            "num_nodes": len(topo.nodes),
            "total_gpus": topo.total_gpus,
            "failure_detected": failure_step >= 0,
            "failure_step": failure_step,
            "time_to_failure_minutes": round(failure_step * self.time_step_seconds / 60, 1) if failure_step >= 0 else None,
            "root_cause": root_cause,
            "timeline": self._event_log,
            "final_snapshot": self.capture_snapshot(),
            "gpu_model": topo.get_gpu(0, 0).spec.model,
        }
