from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import numpy as np

from gpuopt.gpu_monitor import GPUMonitor
from gpuopt.repository import ClusterRepository

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    COLLECTED_AT = 0
    GPU_UTIL = 1
    MEM_UTIL = 2
    TEMP = 3
    POWER = 4
    CLOCK = 5
    ECC_ERR = 6
    RETIRED_PAGES = 7
    XID_ERR = 8
    UTIL_VAR = 9
    TEMP_VAR = 10
    AVAIL_GPUS = 11
    TOTAL_GPUS = 12
    QUEUE_LEN = 13
    JOB_FAILS = 14
    JOB_RETRIES = 15
    AVG_JOB_DUR = 16

    FEATURE_NAMES = [
        "gpu_utilization", "memory_utilization", "temperature", "power_usage",
        "clock_speed", "ecc_errors", "retired_pages", "xid_errors",
        "utilization_variance", "temperature_variance", "available_gpus",
        "total_gpus", "queue_length", "job_failures", "job_retries",
        "average_job_duration",
    ]

    def __init__(
        self,
        repository: ClusterRepository | None = None,
        gpu_monitor: GPUMonitor | None = None,
    ) -> None:
        self._repository = repository
        self._gpu_monitor = gpu_monitor
        self._domain_store: Any = None

    @property
    def domain_store(self) -> Any:
        if self._domain_store is None:
            try:
                from gpuopt.domains.stores import get_domain_store
                self._domain_store = get_domain_store()
            except Exception:
                self._domain_store = None
        return self._domain_store

    def collect_from_gpu_monitor(self) -> list[dict]:
        telemetry_list: list[dict] = []
        if self._gpu_monitor is None:
            return telemetry_list
        try:
            snapshot = self._gpu_monitor.get_snapshot()
            if snapshot and snapshot.devices:
                for dev in snapshot.devices:
                    telemetry_list.append({
                        "gpu_utilization": dev.utilization_gpu_percent,
                        "memory_utilization": dev.utilization_memory_percent,
                        "temperature": dev.temperature_celsius,
                        "power_usage": dev.power_draw_watts,
                        "clock_speed": float(dev.clock_sm_mhz),
                        "ecc_errors": dev.ecc_errors_volatile + dev.ecc_errors_aggregate,
                        "retired_pages": 0,
                        "xid_errors": 0,
                        "utilization_variance": 0.0,
                        "temperature_variance": 0.0,
                        "available_gpus": snapshot.total_gpus - len(snapshot.devices),
                        "total_gpus": snapshot.total_gpus,
                        "queue_length": 0,
                        "job_failures": 0,
                        "job_retries": 0,
                        "average_job_duration": 0.0,
                    })
                logger.info("Collected %d samples from GPU monitor", len(telemetry_list))
        except Exception as exc:
            logger.warning("Failed to collect from GPU monitor: %s", exc)
        return telemetry_list

    def collect_from_domain_store(self, max_samples: int = 500) -> list[dict]:
        telemetry_list: list[dict] = []
        store = self.domain_store
        if store is None:
            return telemetry_list
        try:
            gpu_nodes = store.gpu_node.list(limit=max_samples)
            if gpu_nodes:
                for entry in gpu_nodes:
                    for gpu in entry.gpus:
                        telemetry_list.append({
                            "gpu_utilization": gpu.utilization_gpu_pct,
                            "memory_utilization": gpu.utilization_memory_pct,
                            "temperature": gpu.temperature_gpu_c,
                            "power_usage": gpu.power_watts,
                            "clock_speed": gpu.clock_sm_mhz,
                            "ecc_errors": gpu.ecc_errors_corrected + gpu.ecc_errors_uncorrected,
                            "retired_pages": 0,
                            "xid_errors": gpu.ecc_errors_uncorrected,
                            "utilization_variance": 0.0,
                            "temperature_variance": 0.0,
                            "available_gpus": len(entry.gpus),
                            "total_gpus": len(entry.gpus),
                            "queue_length": 0,
                            "job_failures": 0,
                            "job_retries": 0,
                            "average_job_duration": 0.0,
                        })
                logger.info("Collected %d samples from domain GPU node store", len(telemetry_list))
        except Exception as exc:
            logger.warning("Failed to collect from domain store: %s", exc)

        try:
            scheduler_states = store.scheduler_states.list(limit=max_samples)
            if scheduler_states and telemetry_list:
                avg_queue = np.mean([s.queue_depth for s in scheduler_states if hasattr(s, "queue_depth")])
                avg_wait = np.mean([s.avg_wait_time_seconds for s in scheduler_states if hasattr(s, "avg_wait_time_seconds")])
                for t in telemetry_list:
                    t["queue_length"] = float(avg_queue)
                    t["average_job_duration"] = float(avg_wait)
        except Exception as exc:
            logger.warning("Failed to collect scheduler state: %s", exc)

        return telemetry_list

    def collect_from_repository(self, max_samples: int = 100) -> list[dict]:
        telemetry_list: list[dict] = []
        if self._repository is None:
            return telemetry_list
        try:
            states = self._repository.list_states(limit=max_samples)
            for state in states:
                gpu_devices = [g for n in state.nodes for g in n.gpu_devices]
                if not gpu_devices:
                    continue
                utils = [g.memory_used_bytes / max(g.memory_total_bytes, 1) * 100 for g in gpu_devices]
                total_gpus = len(gpu_devices)
                hot_gpus = sum(1 for u in utils if u > 85)
                telemetry_list.append({
                    "gpu_utilization": float(np.mean(utils)) if utils else 0.0,
                    "memory_utilization": float(np.mean(utils)) if utils else 0.0,
                    "temperature": 0.0,
                    "power_usage": 0.0,
                    "clock_speed": 0.0,
                    "ecc_errors": 0,
                    "retired_pages": 0,
                    "xid_errors": 0,
                    "utilization_variance": float(np.var(utils)) if len(utils) > 1 else 0.0,
                    "temperature_variance": 0.0,
                    "available_gpus": total_gpus - hot_gpus,
                    "total_gpus": total_gpus,
                    "queue_length": 0.0,
                    "job_failures": 0,
                    "job_retries": 0,
                    "average_job_duration": 0.0,
                })
            logger.info("Collected %d samples from repository", len(telemetry_list))
        except Exception as exc:
            logger.warning("Failed to collect from repository: %s", exc)
        return telemetry_list

    def collect_all(self, max_samples: int = 500) -> list[dict]:
        all_data: list[dict] = []
        all_data.extend(self.collect_from_gpu_monitor())
        all_data.extend(self.collect_from_domain_store(max_samples))
        all_data.extend(self.collect_from_repository(max_samples))
        if not all_data:
            logger.warning("No data collected from any source")
        return all_data

    @staticmethod
    def generate_labels(telemetry: dict) -> int:
        risk = 0.0
        count = 0

        ecc = telemetry.get("ecc_errors", 0)
        if ecc > 50:
            risk += 0.4
            count += 1
        elif ecc > 15:
            risk += 0.25
            count += 1

        xid = telemetry.get("xid_errors", 0)
        if xid > 10:
            risk += 0.4
            count += 1
        elif xid > 3:
            risk += 0.2
            count += 1

        temp = telemetry.get("temperature", 0)
        if temp > 85:
            risk += 0.35
            count += 1
        elif temp > 75:
            risk += 0.15
            count += 1

        mem = telemetry.get("memory_utilization", 0)
        if mem > 95:
            risk += 0.40
            count += 1
        elif mem > 85:
            risk += 0.10
            count += 1

        gpu_util = telemetry.get("gpu_utilization", 0)
        if gpu_util > 95:
            risk += 0.1
            count += 1

        avail = telemetry.get("available_gpus", telemetry.get("total_gpus", 1))
        total = telemetry.get("total_gpus", 1)
        if total > 0 and avail < total * 0.1:
            risk += 0.15
            count += 1

        power = telemetry.get("power_usage", 0)
        if power > 400:
            risk += 0.35
            count += 1

        return 1 if risk >= 0.35 else 0

    def build_training_dataset(
        self, max_samples: int = 500
    ) -> tuple[list[dict], list[int]]:
        raw = self.collect_all(max_samples)
        if not raw:
            return [], []
        labels = [self.generate_labels(t) for t in raw]
        logger.info(
            "Built training dataset: %d samples, %d positive (%.1f%%)",
            len(raw), sum(labels), 100 * sum(labels) / max(len(labels), 1),
        )
        return raw, labels


def collect_and_train(
    engine: Any,
    max_samples: int = 500,
    n_synthetic: int = 1000,
    repository: ClusterRepository | None = None,
    gpu_monitor: GPUMonitor | None = None,
) -> dict:
    collector = TrainingDataCollector(
        repository=repository,
        gpu_monitor=gpu_monitor,
    )
    telemetry_data, labels = collector.build_training_dataset(max_samples)
    if telemetry_data:
        logger.info(
            "Training with %d real + %d synthetic samples",
            len(telemetry_data), n_synthetic,
        )
    else:
        logger.info("No real data available, using %d synthetic samples", n_synthetic)
    return engine.train_ensemble(
        telemetry_history=telemetry_data if telemetry_data else None,
        labels=labels if labels else None,
        n_synthetic=n_synthetic,
    )
