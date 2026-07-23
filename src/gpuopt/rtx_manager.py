from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
torch = None
def _get_torch():
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            logger.warning("PyTorch not installed; GPU detection via torch will be unavailable")
            torch = False
            return None
    return torch


@dataclass
class GPUInfo:
    name: str
    index: int
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_percent: float
    temperature_celsius: float
    power_usage_watts: float
    power_limit_watts: float
    driver_version: str
    cuda_version: str
    is_available: bool = True
    ecc_errors: int = 0
    pcie_link_gen: int = 0
    pcie_link_width: int = 0
    partition_id: int | None = None
    physical_gpu_index: int | None = None


@dataclass
class ClusterState:
    timestamp: datetime
    gpus: list[GPUInfo]
    cpu_cores: int
    cpu_usage_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    total_gpu_memory_gb: float
    total_gpu_usage_percent: float
    total_power_watts: float
    active_jobs: list[dict] = field(default_factory=list)


@dataclass
class PartitionSlice:
    id: int
    gb: float
    node_name: str


class RTXClusterManager:

    def __init__(self) -> None:
        self.cluster_id = "rtx4090-cluster"
        self.cluster_name = "RTX 4090 AI Cluster"
        self.jobs: list[dict] = []
        self.job_counter = 0
        self.partitions: list[PartitionSlice] = []
        self._load_partitions()

    def _load_partitions(self) -> None:
        raw = os.environ.get("GPUOPT_RTX_PARTITIONS_GB", "").strip()
        if not raw:
            return
        for i, part in enumerate(raw.split(",")):
            part = part.strip()
            if not part:
                continue
            try:
                gb = float(part)
                self.partitions.append(PartitionSlice(id=i + 1, gb=gb, node_name=f"node{i + 1}"))
            except ValueError:
                logger.warning("Invalid partition size: %s", part)

    def _detect_via_nvidia_smi(self) -> list[GPUInfo]:
        gpus: list[GPUInfo] = []
        queries = [
            "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,driver_version",
            "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit",
        ]
        output = None
        for q in queries:
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                    universal_newlines=True, timeout=5,
                )
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        if output is None:
            logger.warning("nvidia-smi not available")
            return gpus
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(", ")]
            if len(parts) < 8:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                idx = 0
            name = parts[1]
            try:
                mem_total = float(parts[2].replace(" MiB", "")) / 1024
                mem_used = float(parts[3].replace(" MiB", "")) / 1024
            except ValueError:
                mem_total = 0
                mem_used = 0
            try:
                util = float(parts[4].replace(" %", ""))
            except ValueError:
                util = 0
            try:
                temp = float(parts[5].replace(" °C", ""))
            except ValueError:
                temp = 0
            try:
                power_draw = float(parts[6].replace(" W", ""))
            except ValueError:
                power_draw = 0
            try:
                power_limit = float(parts[7].replace(" W", ""))
            except ValueError:
                power_limit = 0
            driver = parts[8] if len(parts) > 8 else "Unknown"
            gpus.append(GPUInfo(
                name=name, index=idx,
                memory_total_gb=mem_total, memory_used_gb=mem_used,
                memory_free_gb=mem_total - mem_used,
                utilization_percent=util, temperature_celsius=temp,
                power_usage_watts=power_draw, power_limit_watts=power_limit,
                driver_version=driver, cuda_version="",
            ))
        return gpus

    def detect_gpus(self) -> list[GPUInfo]:
        physical_gpus: list[GPUInfo] = []
        cuda_version = "Unknown"
        try:
            _torch = _get_torch()
            if _torch and _torch.cuda.is_available():
                for i in range(_torch.cuda.device_count()):
                    props = _torch.cuda.get_device_properties(i)

                    driver = "Unknown"
                    pcie_gen = 0
                    pcie_width = 0
                    ecc = 0
                    power_draw = 0.0
                    power_limit = 0.0
                    temp = 0.0
                    util = 0.0
                    mem_used_smi = 0.0
                    mem_total_smi = props.total_memory / 1024**3

                    try:
                        output = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,driver_version",
                                "--format=csv,noheader",
                                "--id=" + str(i),
                            ],
                            universal_newlines=True,
                            timeout=5,
                        ).strip()
                        parts = [p.strip() for p in output.split(", ")]
                        if len(parts) >= 8:
                            driver = parts[8]
                            try:
                                mem_total_smi = float(parts[1].replace(" MiB", "")) / 1024
                                mem_used_smi = float(parts[2].replace(" MiB", "")) / 1024
                            except ValueError:
                                pass
                            try:
                                util = float(parts[4].replace(" %", ""))
                            except ValueError:
                                pass
                            try:
                                temp = float(parts[5].replace(" °C", ""))
                            except ValueError:
                                pass
                            try:
                                power_draw = float(parts[6].replace(" W", ""))
                            except ValueError:
                                pass
                            try:
                                power_limit = float(parts[7].replace(" W", ""))
                            except ValueError:
                                pass
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                        pass

                    physical_gpus.append(GPUInfo(
                        name=props.name,
                        index=i,
                        memory_total_gb=mem_total_smi,
                        memory_used_gb=mem_used_smi,
                        memory_free_gb=mem_total_smi - mem_used_smi,
                        utilization_percent=util,
                        temperature_celsius=temp,
                        power_usage_watts=power_draw,
                        power_limit_watts=power_limit,
                        driver_version=driver,
                        cuda_version=_torch.version.cuda or "Unknown",
                    ))
                cuda_version = _torch.version.cuda or "Unknown"
        except Exception as exc:
            logger.warning("PyTorch GPU detection failed: %s", exc)

        if not physical_gpus:
            physical_gpus = self._detect_via_nvidia_smi()
            if physical_gpus:
                logger.info("Detected %d GPU(s) via nvidia-smi fallback", len(physical_gpus))

        if not physical_gpus:
            physical_gpus.append(GPUInfo(
                name="No GPU Detected",
                index=0,
                memory_total_gb=0,
                memory_used_gb=0,
                memory_free_gb=0,
                utilization_percent=0,
                temperature_celsius=0,
                power_usage_watts=0,
                power_limit_watts=0,
                driver_version="N/A",
                cuda_version="N/A",
                is_available=False,
            ))

        if not self.partitions or not physical_gpus[0].is_available:
            return physical_gpus

        virtual_gpus: list[GPUInfo] = []
        for phys in physical_gpus:
            for part in self.partitions:
                mem_total = part.gb
                mem_used = min(phys.memory_used_gb * (part.gb / phys.memory_total_gb), part.gb * 0.9) if phys.memory_total_gb > 0 else 0
                mem_free = mem_total - mem_used
                util = min(phys.utilization_percent, 100.0)
                power = phys.power_usage_watts * (part.gb / phys.memory_total_gb) if phys.memory_total_gb > 0 else 0
                virtual_gpus.append(GPUInfo(
                    name=f"{phys.name} [Part {part.id}: {part.node_name}]",
                    index=len(virtual_gpus),
                    memory_total_gb=mem_total,
                    memory_used_gb=mem_used,
                    memory_free_gb=mem_free,
                    utilization_percent=util,
                    temperature_celsius=phys.temperature_celsius,
                    power_usage_watts=power,
                    power_limit_watts=phys.power_limit_watts * (part.gb / phys.memory_total_gb) if phys.memory_total_gb > 0 else 0,
                    driver_version=phys.driver_version,
                    cuda_version=phys.cuda_version or cuda_version,
                    ecc_errors=phys.ecc_errors,
                    pcie_link_gen=phys.pcie_link_gen,
                    pcie_link_width=phys.pcie_link_width,
                    partition_id=part.id,
                    physical_gpu_index=phys.index,
                ))
        return virtual_gpus

    def nvidia_smi_query(self, query: str) -> list[str]:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                universal_newlines=True,
                timeout=5,
            )
            return [line.strip() for line in output.strip().split("\n") if line.strip()]
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return []

    def get_cluster_state(self) -> ClusterState:
        gpus = self.detect_gpus()
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / 1024**3
        memory_used_gb = memory.used / 1024**3
        memory_free_gb = memory.available / 1024**3

        total_memory = sum(g.memory_total_gb for g in gpus)
        total_usage = sum(g.utilization_percent for g in gpus) / max(len(gpus), 1)
        total_power = sum(g.power_usage_watts for g in gpus)

        return ClusterState(
            timestamp=datetime.now(timezone.utc),
            gpus=gpus,
            cpu_cores=cpu_cores,
            cpu_usage_percent=cpu_usage,
            memory_total_gb=memory_total_gb,
            memory_used_gb=memory_used_gb,
            memory_free_gb=memory_free_gb,
            total_gpu_memory_gb=total_memory,
            total_gpu_usage_percent=total_usage,
            total_power_watts=total_power,
            active_jobs=self.jobs,
        )

    def submit_job(self, job_spec: dict) -> dict:
        self.job_counter += 1
        job_id = f"job-{self.job_counter:04d}"

        job: dict[str, Any] = {
            "job_id": job_id,
            "name": job_spec.get("name", f"Job {self.job_counter}"),
            "required_gpus": job_spec.get("required_gpus", 1),
            "required_memory_gb": job_spec.get("required_memory_gb", 8.0),
            "estimated_runtime_hours": job_spec.get("estimated_runtime_hours", 1.0),
            "priority": job_spec.get("priority", 5),
            "status": "submitted",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "assigned_gpu": None,
            "assigned_gpu_names": None,
        }

        state = self.get_cluster_state()
        available_gpus = [g for g in state.gpus if g.memory_free_gb >= job["required_memory_gb"]]

        if len(available_gpus) >= job["required_gpus"] and state.memory_free_gb >= job["required_memory_gb"]:
            assigned = available_gpus[: job["required_gpus"]]
            job["status"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            job["assigned_gpu"] = [g.index for g in assigned]
            job["assigned_gpu_names"] = [g.name for g in assigned]
            self.jobs.append(job)
            return {
                "success": True,
                "job": job,
                "message": f"Job {job_id} assigned to {len(assigned)} GPU(s): {', '.join(job['assigned_gpu_names'])}",
            }

        job["status"] = "queued"
        self.jobs.append(job)
        return {
            "success": False,
            "job": job,
            "message": (
                f"Insufficient resources. "
                f"Requested: {job['required_gpus']} GPU(s), {job['required_memory_gb']} GB. "
                f"Available: {len(available_gpus)} GPU(s), {state.memory_free_gb:.1f} GB system RAM"
            ),
        }

    def get_job_status(self, job_id: str) -> Optional[dict]:
        for job in self.jobs:
            if job["job_id"] == job_id:
                return job
        return None

    def list_jobs(self) -> list[dict]:
        return self.jobs

    def simulate_job(self, job_spec: dict) -> dict:
        state = self.get_cluster_state()
        candidate_gpus = [g for g in state.gpus if g.memory_free_gb >= job_spec.get("required_memory_gb", 8.0)]
        required = job_spec.get("required_gpus", 1)
        feasible = len(candidate_gpus) >= required
        success_rate = 0.95 if feasible else max(0.1, len(candidate_gpus) / max(required, 1))

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_spec": job_spec,
            "cluster_capacity": {
                "total_gpus": len(state.gpus),
                "available_gpus": len(candidate_gpus),
                "total_gpu_memory_gb": state.total_gpu_memory_gb,
                "free_gpu_memory_gb": sum(g.memory_free_gb for g in state.gpus),
                "free_system_memory_gb": state.memory_free_gb,
            },
            "simulation": {
                "feasible": feasible,
                "candidate_gpus": [g.index for g in candidate_gpus[:required]],
                "predicted_runtime_hours": job_spec.get("estimated_runtime_hours", 1.0),
                "predicted_success_rate": round(success_rate, 2),
                "estimated_power_cost_kwh": round(
                    sum(g.power_usage_watts for g in candidate_gpus[:required])
                    * job_spec.get("estimated_runtime_hours", 1.0)
                    / 1000,
                    2,
                ),
            },
        }


manager = RTXClusterManager()