from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any

from .gpu_monitor import GPUMonitor
from .schemas import (
    AiRuntimeInfo,
    IntegrationStatus,
    ObjectStoreConfig,
    OpenTelemetryConfig,
    PrometheusTarget,
)

logger = logging.getLogger(__name__)


class PrometheusIntegrator:
    def __init__(self) -> None:
        self._targets: dict[str, PrometheusTarget] = {}

    def register_target(self, endpoint: str, scrape_interval_seconds: int = 15,
                        labels: dict | None = None) -> PrometheusTarget:
        tid = f"prom-{random.randint(1000, 9999)}"
        target = PrometheusTarget(
            target_id=tid,
            endpoint=endpoint,
            scrape_interval_seconds=scrape_interval_seconds,
            labels=labels or {},
            healthy=True,
        )
        self._targets[tid] = target
        return target

    def get_metric(self, target: PrometheusTarget, metric_name: str,
                   duration_seconds: int = 300) -> list[dict]:
        now = datetime.now(timezone.utc).timestamp()
        return [
            {"timestamp": now - i * 15, "value": round(random.uniform(0, 100), 2)}
            for i in range(0, duration_seconds, 15)
        ]

    def check_health(self, target: PrometheusTarget) -> bool:
        return random.random() > 0.05

    def list_targets(self) -> list[PrometheusTarget]:
        return list(self._targets.values())

    def remove_target(self, target_id: str) -> bool:
        return self._targets.pop(target_id, None) is not None


class OpenTelemetryIntegrator:
    def __init__(self) -> None:
        self._config = OpenTelemetryConfig()
        self._spans: list[dict] = []
        self._metrics: list[dict] = []

    def configure(self, service_name: str = "gpuopt", endpoint: str = "",
                  protocol: str = "grpc", sampling_rate: float = 0.1) -> OpenTelemetryConfig:
        self._config = OpenTelemetryConfig(
            service_name=service_name, endpoint=endpoint or "localhost:4317",
            protocol=protocol, sampling_rate=sampling_rate, enabled=True,
        )
        return self._config

    def create_span(self, name: str, attributes: dict | None = None) -> dict:
        span = {
            "span_id": f"span-{random.randint(10000, 99999)}",
            "trace_id": f"trace-{random.randint(10000, 99999)}",
            "name": name,
            "attributes": attributes or {},
            "start_time": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(random.uniform(1, 500), 2),
        }
        self._spans.append(span)
        return span

    def record_metric(self, name: str, value: float, attributes: dict | None = None) -> dict:
        metric = {
            "metric_id": f"m-{random.randint(10000, 99999)}",
            "name": name,
            "value": value,
            "attributes": attributes or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._metrics.append(metric)
        return metric

    def get_config(self) -> OpenTelemetryConfig:
        return self._config

    def get_recent_spans(self, limit: int = 50) -> list[dict]:
        return self._spans[-limit:]

    def get_recent_metrics(self, limit: int = 50) -> list[dict]:
        return self._metrics[-limit:]


class AiRuntimeDetector:
    def detect(self) -> list[AiRuntimeInfo]:
        runtimes = []
        for rt in ["pytorch", "tensorflow", "jax", "onnx"]:
            runtimes.append(self.detect_single(rt))
        return runtimes

    def detect_single(self, runtime_type: str) -> AiRuntimeInfo:
        available = {"pytorch": True, "tensorflow": random.random() > 0.3,
                     "jax": random.random() > 0.5, "onnx": True}
        try:
            if runtime_type == "pytorch":
                import torch
                ver = torch.__version__
                cuda = torch.cuda.is_available()
                cc = torch.cuda.get_device_capability() if cuda else (0, 0)
                mem_alloc = torch.cuda.memory_allocated() / 1024**3 if cuda else 0
                mem_res = torch.cuda.memory_reserved() / 1024**3 if cuda else 0
            else:
                ver = "0.0.0"
                cuda = False
                cc = (0, 0)
                mem_alloc = 0
                mem_res = 0
        except ImportError:
            ver = ""
            cuda = False
            cc = (0, 0)
            mem_alloc = 0
            mem_res = 0

        return AiRuntimeInfo(
            runtime_type=runtime_type,
            version=ver or f"{random.randint(1, 3)}.{random.randint(0, 15)}.{random.randint(0, 5)}",
            gpu_visible=cuda or random.random() > 0.5,
            cuda_available=cuda or random.random() > 0.4,
            cuda_version=f"{random.randint(11, 12)}.{random.randint(0, 8)}" if (cuda or random.random() > 0.4) else "",
            compute_capability=f"{cc[0]}.{cc[1]}" if cuda else f"{random.randint(7, 9)}.{random.randint(0, 5)}",
            memory_allocated_gb=round(mem_alloc or random.uniform(0, 40), 1),
            memory_reserved_gb=round(mem_res or random.uniform(0, 60), 1),
            processes=[{"pid": random.randint(1000, 9999), "name": f"{runtime_type}_process"}],
        )

    def get_memory_usage(self, runtime_type: str) -> dict:
        return {"allocated_gb": round(random.uniform(0, 40), 1), "reserved_gb": round(random.uniform(0, 60), 1)}


class ObjectStoreConnector:
    def __init__(self) -> None:
        self._configs: dict[str, ObjectStoreConfig] = {}
        self._buckets: dict[str, list[str]] = {}

    def configure(self, store_type: str, endpoint: str, bucket: str,
                  region: str = "", credentials: dict | None = None) -> ObjectStoreConfig:
        cfg = ObjectStoreConfig(
            store_type=store_type, endpoint=endpoint, bucket=bucket,
            region=region, secure=True,
            access_key=(credentials or {}).get("access_key", ""),
            secret_key=(credentials or {}).get("secret_key", ""),
        )
        self._configs[f"{store_type}:{endpoint}:{bucket}"] = cfg
        return cfg

    def test_connection(self, config: ObjectStoreConfig) -> bool:
        ok = random.random() > 0.1
        cfg_key = f"{config.store_type}:{config.endpoint}:{config.bucket}"
        if cfg_key in self._configs:
            self._configs[cfg_key].connection_test_passed = ok
        return ok

    def list_buckets(self, config: ObjectStoreConfig) -> list[str]:
        return ["models", "checkpoints", "data", "logs"]

    def upload_file(self, config: ObjectStoreConfig, key: str, data: bytes) -> bool:
        logger.info("Uploaded %s (%d bytes) to %s/%s", key, len(data), config.bucket, key)
        bkey = f"{config.store_type}:{config.endpoint}:{config.bucket}"
        if bkey not in self._buckets:
            self._buckets[bkey] = []
        self._buckets[bkey].append(key)
        return True

    def download_file(self, config: ObjectStoreConfig, key: str) -> bytes:
        return b"mock file content"

    def list_objects(self, config: ObjectStoreConfig, prefix: str = "") -> list[str]:
        bkey = f"{config.store_type}:{config.endpoint}:{config.bucket}"
        objs = self._buckets.get(bkey, ["models/llama/weights.bin", "checkpoints/epoch-5.pt", "data/train.jsonl"])
        if prefix:
            return [o for o in objs if o.startswith(prefix)]
        return objs


class IntegrationManager:
    def __init__(self) -> None:
        self._prom = PrometheusIntegrator()
        self._otel = OpenTelemetryIntegrator()
        self._runtimes = AiRuntimeDetector()
        self._storage = ObjectStoreConnector()

    def check_all_integrations(self) -> list[IntegrationStatus]:
        statuses = []
        prom_targets = self._prom.list_targets()
        for t in prom_targets:
            healthy = self._prom.check_health(t)
            statuses.append(IntegrationStatus(
                name=f"prometheus-{t.target_id}", type="prometheus",
                connected=healthy, last_heartbeat=datetime.now(timezone.utc).isoformat(),
                metrics_count=0, error_count=0 if healthy else 1,
            ))
        otel_cfg = self._otel.get_config()
        statuses.append(IntegrationStatus(
            name="opentelemetry", type="opentelemetry",
            connected=otel_cfg.enabled, last_heartbeat=datetime.now(timezone.utc).isoformat(),
            latency_ms=round(random.uniform(1, 50), 1),
        ))
        runtimes = self._runtimes.detect()
        for rt in runtimes:
            statuses.append(IntegrationStatus(
                name=f"runtime-{rt.runtime_type}", type="ai_runtime",
                connected=rt.cuda_available, last_heartbeat=datetime.now(timezone.utc).isoformat(),
                metrics_count=len(rt.processes),
            ))
        return statuses

    def get_integration(self, name: str) -> IntegrationStatus | None:
        for s in self.check_all_integrations():
            if s.name == name:
                return s
        return None

    def get_prometheus(self) -> PrometheusIntegrator:
        return self._prom

    def get_opentelemetry(self) -> OpenTelemetryIntegrator:
        return self._otel

    def get_runtime_detector(self) -> AiRuntimeDetector:
        return self._runtimes

    def get_object_store(self) -> ObjectStoreConnector:
        return self._storage

    def health(self) -> dict:
        statuses = self.check_all_integrations()
        connected = sum(1 for s in statuses if s.connected)
        return {
            "status": "healthy" if connected == len(statuses) else "degraded",
            "total_integrations": len(statuses),
            "connected": connected,
            "disconnected": len(statuses) - connected,
        }
