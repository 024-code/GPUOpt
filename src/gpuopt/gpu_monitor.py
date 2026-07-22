from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_HAS_PYNVML = False
try:
    import pynvml  # noqa: F401
    _HAS_PYNVML = True
except ImportError:
    pass


@dataclass
class GPUProcessInfo:
    pid: int = 0
    process_name: str = ""
    used_gpu_memory_mb: int = 0
    gpu_index: int = 0
    gpu_utilization: float = 0.0


@dataclass
class GPUDeviceSnapshot:
    index: int = 0
    uuid: str = ""
    model: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    utilization_gpu_percent: float = 0.0
    utilization_memory_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    power_limit_watts: float = 0.0
    fan_speed_percent: float = 0.0
    pcie_link_gen: int = 0
    pcie_link_width: int = 0
    clock_sm_mhz: int = 0
    clock_mem_mhz: int = 0
    ecc_errors_volatile: int = 0
    ecc_errors_aggregate: int = 0
    processes: list[GPUProcessInfo] = field(default_factory=list)


@dataclass
class GPUSnapshot:
    collected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hostname: str = ""
    devices: list[GPUDeviceSnapshot] = field(default_factory=list)
    driver_version: str = ""
    cuda_version: str = ""
    total_gpus: int = 0
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    free_memory_mb: int = 0


class GPUMonitor:
    def __init__(self, poll_interval: float = 15.0) -> None:
        self._poll_interval = poll_interval
        self._snapshot: GPUSnapshot | None = None
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._use_nvml = _HAS_PYNVML

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="gpu-monitor")
        self._thread.start()
        logger.info("GPU monitor started (interval=%ss, nvml=%s)", self._poll_interval, self._use_nvml)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("GPU monitor stopped")

    def _poll_loop(self) -> None:
        while self._running:
            try:
                snapshot = self.collect()
                with self._lock:
                    self._snapshot = snapshot
            except Exception as exc:
                logger.error("GPU monitor poll error: %s", exc)
            threading.Event().wait(self._poll_interval)

    def collect(self) -> GPUSnapshot:
        if self._use_nvml:
            return self._collect_nvml()
        return self._collect_nvidia_smi()

    def get_snapshot(self) -> GPUSnapshot | None:
        with self._lock:
            return self._snapshot

    def _collect_nvml(self) -> GPUSnapshot:
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            devices: list[GPUDeviceSnapshot] = []
            total_mem = 0
            used_mem = 0
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except Exception:
                    fan = 0
                try:
                    pcie = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                    pcie_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                except Exception:
                    pcie = 0
                    pcie_width = 0
                try:
                    clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    clock_sm = 0
                    clock_mem = 0
                try:
                    ecc_vol = pynvml.nvmlDeviceGetTotalEccErrors(handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_VOLATILE_ECC)
                    ecc_agg = pynvml.nvmlDeviceGetTotalEccErrors(handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_AGGREGATE_ECC)
                except Exception:
                    ecc_vol = 0
                    ecc_agg = 0

                procs: list[GPUProcessInfo] = []
                try:
                    running_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in running_procs:
                        try:
                            proc_name = pynvml.nvmlSystemGetProcessName(p.pid) or ""
                        except Exception:
                            proc_name = ""
                        procs.append(GPUProcessInfo(
                            pid=p.pid, process_name=proc_name or f"pid-{p.pid}",
                            used_gpu_memory_mb=p.usedGpuMemory // (1024 * 1024),
                            gpu_index=i,
                        ))
                except Exception:
                    pass

                total_mem += mem.total
                used_mem += mem.used
                devices.append(GPUDeviceSnapshot(
                    index=i, uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
                    model=name.decode() if isinstance(name, bytes) else name,
                    memory_total_mb=mem.total // (1024 * 1024),
                    memory_used_mb=mem.used // (1024 * 1024),
                    memory_free_mb=(mem.total - mem.used) // (1024 * 1024),
                    utilization_gpu_percent=float(util.gpu),
                    utilization_memory_percent=float(util.memory),
                    temperature_celsius=float(temp),
                    power_draw_watts=power / 1000.0,
                    power_limit_watts=power_limit / 1000.0 if power_limit else 0,
                    fan_speed_percent=float(fan),
                    pcie_link_gen=pcie,
                    pcie_link_width=pcie_width,
                    clock_sm_mhz=clock_sm,
                    clock_mem_mhz=clock_mem,
                    ecc_errors_volatile=ecc_vol,
                    ecc_errors_aggregate=ecc_agg,
                    processes=procs,
                ))
            driver = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
            return GPUSnapshot(
                hostname="",
                devices=devices, driver_version=driver.decode() if isinstance(driver, bytes) else driver,
                total_gpus=count, total_memory_mb=total_mem // (1024 * 1024),
                used_memory_mb=used_mem // (1024 * 1024),
                free_memory_mb=(total_mem - used_mem) // (1024 * 1024),
            )
        except Exception as exc:
            logger.debug("NVML collection failed: %s", exc)
            self._use_nvml = False
            return self._collect_nvidia_smi()

    def _collect_nvidia_smi(self) -> GPUSnapshot:
        try:
            result = subprocess.run(
                shlex.split("nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
                            "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,"
                            "fan.speed,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem,"
                            "ecc.errors.volatile.device_memory,ecc.errors.aggregate.device_memory "
                            "--format=csv,noheader,nounits"),
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return self._mock_snapshot()
            devices: list[GPUDeviceSnapshot] = []
            total_mem = 0
            used_mem = 0
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 13:
                    continue
                try:
                    idx = int(parts[0])
                    uuid_val = parts[1]
                    model = parts[2]
                    mem_total = int(float(parts[3])) if parts[3] else 0
                    mem_used = int(float(parts[4])) if parts[4] else 0
                    mem_free = int(float(parts[5])) if parts[5] else 0
                    gpu_util = float(parts[6]) if parts[6] else 0.0
                    mem_util = float(parts[7]) if parts[7] else 0.0
                    temp = float(parts[8]) if parts[8] else 0.0
                    power = float(parts[9]) if parts[9] else 0.0
                    power_limit = float(parts[10]) if parts[10] else 0.0
                    fan = float(parts[11]) if parts[11] else 0.0
                    pcie_gen = int(float(parts[12])) if parts[12] else 0
                    pcie_width = int(float(parts[13])) if parts[13] else 0
                    clock_sm = int(float(parts[14])) if len(parts) > 14 and parts[14] else 0
                    clock_mem = int(float(parts[15])) if len(parts) > 15 and parts[15] else 0
                    ecc_vol = int(float(parts[16])) if len(parts) > 16 and parts[16] else 0
                    ecc_agg = int(float(parts[17])) if len(parts) > 17 and parts[17] else 0
                except (ValueError, IndexError):
                    continue

                total_mem += mem_total
                used_mem += mem_used
                devices.append(GPUDeviceSnapshot(
                    index=idx, uuid=uuid_val, model=model,
                    memory_total_mb=mem_total, memory_used_mb=mem_used, memory_free_mb=mem_free,
                    utilization_gpu_percent=gpu_util, utilization_memory_percent=mem_util,
                    temperature_celsius=temp, power_draw_watts=power, power_limit_watts=power_limit,
                    fan_speed_percent=fan, pcie_link_gen=pcie_gen, pcie_link_width=pcie_width,
                    clock_sm_mhz=clock_sm, clock_mem_mhz=clock_mem,
                    ecc_errors_volatile=ecc_vol, ecc_errors_aggregate=ecc_agg,
                ))

            procs_output = subprocess.run(
                shlex.split("nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_index --format=csv,noheader,nounits"),
                capture_output=True, text=True, timeout=15,
            )
            if procs_output.returncode == 0:
                for line in procs_output.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        try:
                            pid = int(parts[0])
                            name = parts[1]
                            mem = int(float(parts[2])) if parts[2] else 0
                            gpu_idx = int(parts[3])
                            for dev in devices:
                                if dev.index == gpu_idx:
                                    dev.processes.append(GPUProcessInfo(pid=pid, process_name=name, used_gpu_memory_mb=mem, gpu_index=gpu_idx))
                        except (ValueError, IndexError):
                            pass

            return GPUSnapshot(
                devices=devices, total_gpus=len(devices),
                total_memory_mb=total_mem, used_memory_mb=used_mem,
                free_memory_mb=total_mem - used_mem,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            logger.debug("nvidia-smi not available: %s", exc)
            return self._mock_snapshot()

    def _mock_snapshot(self) -> GPUSnapshot:
        import random
        devices = [
            GPUDeviceSnapshot(
                index=i, uuid=f"mock-gpu-{i}", model=f"mock-{['H100','A100','V100'][i % 3]}",
                memory_total_mb=81920, memory_used_mb=int(random.uniform(10000, 70000)),
                memory_free_mb=0,
                utilization_gpu_percent=random.uniform(10, 95),
                utilization_memory_percent=random.uniform(20, 90),
                temperature_celsius=random.uniform(40, 85),
                power_draw_watts=random.uniform(100, 400),
                power_limit_watts=350,
                processes=[
                    GPUProcessInfo(pid=1000 + i, process_name=f"train-job-{i}", used_gpu_memory_mb=int(random.uniform(1000, 32000)), gpu_index=i)
                ],
            )
            for i in range(random.randint(1, 4))
        ]
        for d in devices:
            d.memory_free_mb = d.memory_total_mb - d.memory_used_mb
        total = sum(d.memory_total_mb for d in devices)
        used = sum(d.memory_used_mb for d in devices)
        return GPUSnapshot(
            devices=devices, total_gpus=len(devices),
            total_memory_mb=total, used_memory_mb=used, free_memory_mb=total - used,
        )
