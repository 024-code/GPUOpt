from __future__ import annotations

import logging
import random
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .gpu_monitor import GPUMonitor
from .schemas import (
    EndpointTelemetry,
    FabricLinkTelemetry,
    FabricTelemetry,
    JobTelemetry,
    ModelServiceTelemetry,
    QueueTelemetry,
    TelemetrySnapshot,
    TelemetryStreamEvent,
)

logger = logging.getLogger(__name__)


class ModelServiceCollector:
    def collect(self, model_name: str = "llama", num_endpoints: int = 2) -> ModelServiceTelemetry:
        endpoints = []
        for i in range(num_endpoints):
            total_req = random.randint(1000, 50000)
            avg_lat = random.uniform(10.0, 200.0)
            endpoints.append(EndpointTelemetry(
                endpoint=f"endpoint-{i}",
                total_requests=total_req,
                requests_per_second=round(total_req / 3600, 2),
                avg_latency_ms=round(avg_lat, 1),
                p50_latency_ms=round(avg_lat * 0.8, 1),
                p99_latency_ms=round(avg_lat * 2.5, 1),
                error_count=random.randint(0, 50),
                error_rate=round(random.uniform(0.0, 0.05), 4),
                throughput_tokens_per_sec=round(random.uniform(50, 500), 1),
            ))
        return ModelServiceTelemetry(
            model_name=model_name,
            model_version=f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            replicas=random.randint(1, 8),
            endpoints=endpoints,
            avg_gpu_utilization=round(random.uniform(20, 95), 1),
            avg_memory_utilization=round(random.uniform(30, 90), 1),
            cpu_usage_percent=round(random.uniform(10, 80), 1),
        )


class FabricTelemetryCollector:
    def collect(self, node: str = "node-0", gpu_index: int = 0, num_links: int = 4) -> FabricTelemetry:
        links = []
        for i in range(num_links):
            lt = random.choice(["nvlink", "pcie", "nvswitch"])
            links.append(FabricLinkTelemetry(
                link_index=i,
                link_type=lt,
                bandwidth_usage_percent=round(random.uniform(0, 100), 1),
                tx_bytes_per_sec=round(random.uniform(1e6, 5e10), 0),
                rx_bytes_per_sec=round(random.uniform(1e6, 5e10), 0),
                errors=random.randint(0, 10),
                crc_errors=random.randint(0, 3),
                link_width=random.choice([4, 8, 16]),
                link_gen=random.randint(2, 5),
                is_active=True,
            ))
        return FabricTelemetry(
            node=node,
            gpu_index=gpu_index,
            links=links,
            nvlink_bandwidth_utilization=round(random.uniform(0, 95), 1),
            pcie_bandwidth_utilization=round(random.uniform(0, 60), 1),
            total_nvlink_errors=sum(l.errors for l in links),
        )


class QueueTelemetryCollector:
    def collect(self, queue_name: str = "default", num_queues: int = 3) -> list[QueueTelemetry]:
        queues = []
        for i in range(num_queues):
            qn = f"{queue_name}-{i}" if num_queues > 1 else queue_name
            depth = random.randint(0, 50)
            pending = random.randint(0, depth)
            running = random.randint(1, 20)
            queues.append(QueueTelemetry(
                queue_name=qn,
                queue_depth=depth,
                pending_jobs=pending,
                running_jobs=running,
                avg_wait_time_seconds=round(random.uniform(10, 3600), 1),
                max_wait_time_seconds=round(random.uniform(100, 7200), 1),
                p99_wait_time_seconds=round(random.uniform(60, 6000), 1),
                submission_rate_per_minute=round(random.uniform(0.1, 10), 2),
                completion_rate_per_minute=round(random.uniform(0.1, 9), 2),
                backlog_growth_rate=round(random.uniform(-2, 5), 2),
                priority_breaks=random.randint(0, 5),
                starved_jobs=random.randint(0, 3),
            ))
        return queues


class JobTelemetryCollector:
    def collect(self, num_jobs: int = 5) -> list[JobTelemetry]:
        jobs = []
        states = ["running", "completed", "queued", "failed", "preempted"]
        for i in range(num_jobs):
            state = random.choice(states)
            oom = state == "failed" and random.random() < 0.3
            preempted = state == "preempted"
            wait = random.uniform(0, 3600)
            jobs.append(JobTelemetry(
                job_id=f"job-{uuid4().hex[:8]}",
                job_name=f"training-job-{i}",
                state=state,
                priority=random.randint(1, 10),
                gpu_required=random.randint(1, 8),
                memory_required_gb=round(random.uniform(4, 80), 1),
                wall_time_seconds=round(random.uniform(100, 86400), 0),
                wait_time_seconds=round(wait, 0),
                gpu_utilization_avg=round(random.uniform(10, 98), 1),
                memory_utilization_avg=round(random.uniform(20, 95), 1),
                oom_killed=oom,
                exit_code=-9 if oom else (0 if state == "completed" else 1),
                preempted=preempted,
                submitted_at=datetime.now(timezone.utc).isoformat(),
                started_at=datetime.now(timezone.utc).isoformat() if state != "queued" else "",
                completed_at=datetime.now(timezone.utc).isoformat() if state in ("completed", "failed") else "",
            ))
        return jobs


class ExtendedTelemetryService:
    def __init__(self) -> None:
        self._monitor = GPUMonitor()
        self._model_collector = ModelServiceCollector()
        self._fabric_collector = FabricTelemetryCollector()
        self._queue_collector = QueueTelemetryCollector()
        self._job_collector = JobTelemetryCollector()

    def collect_full_snapshot(self, cluster_id: str = "") -> TelemetrySnapshot:
        try:
            gpu_snap = self._monitor.collect()
            gpu_dict = {
                "total_gpus": gpu_snap.total_gpus,
                "total_memory_mb": gpu_snap.total_memory_mb,
                "used_memory_mb": gpu_snap.used_memory_mb,
                "devices": [
                    {
                        "index": d.index, "model": d.model,
                        "memory_used_mb": d.memory_used_mb,
                        "utilization_percent": d.utilization_gpu_percent,
                        "temperature_celsius": d.temperature_celsius,
                    }
                    for d in gpu_snap.devices
                ],
            }
        except Exception as exc:
            logger.debug("GPU snapshot failed: %s", exc)
            gpu_dict = {}

        models = [self._model_collector.collect(f"model-{i}") for i in range(random.randint(1, 3))]
        fabrics = [self._fabric_collector.collect(f"node-{n}", g)
                   for n in range(random.randint(1, 3))
                   for g in range(random.randint(1, 4))]
        queues = self._queue_collector.collect()
        jobs = self._job_collector.collect(random.randint(3, 8))

        return TelemetrySnapshot(
            cluster_id=cluster_id,
            gpu_snapshot=gpu_dict,
            model_services=models,
            fabric=fabrics,
            queues=queues,
            jobs=jobs,
        )

    def stream_events(self, cluster_id: str = "") -> list[TelemetryStreamEvent]:
        event_types = ["gpu", "model_service", "fabric", "queue", "job"]
        events = []
        for et in event_types:
            events.append(TelemetryStreamEvent(
                event_type=et,
                cluster_id=cluster_id,
                data={"sample": True, "timestamp": datetime.now(timezone.utc).isoformat()},
            ))
        return events


class TelemetryStreamer:
    def __init__(self, buffer_size: int = 1000) -> None:
        self._buffer: list[TelemetryStreamEvent] = []
        self._lock = threading.Lock()
        self._buffer_size = buffer_size
        self._running = False
        self._thread: threading.Thread | None = None
        self._service = ExtendedTelemetryService()

    def start(self, interval_seconds: float = 5.0) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, args=(interval_seconds,), daemon=True)
        self._thread.start()
        logger.info("Telemetry streamer started (interval=%ss)", interval_seconds)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Telemetry streamer stopped")

    def get_events(self, event_type: str | None = None, limit: int = 100) -> list[TelemetryStreamEvent]:
        with self._lock:
            events = self._buffer
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return events[-limit:]

    def _run_loop(self, interval: float) -> None:
        while self._running:
            try:
                events = self._service.stream_events()
                with self._lock:
                    self._buffer.extend(events)
                    if len(self._buffer) > self._buffer_size:
                        self._buffer = self._buffer[-self._buffer_size:]
            except Exception as exc:
                logger.error("Streamer error: %s", exc)
            threading.Event().wait(interval)


def uuid4() -> Any:
    import uuid
    return uuid.uuid4()
