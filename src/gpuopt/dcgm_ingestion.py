from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class DcgmMetricType(str, Enum):
    GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
    MEM_COPY_UTIL = "DCGM_FI_DEV_MEM_COPY_UTIL"
    ENC_UTIL = "DCGM_FI_DEV_ENC_UTIL"
    DEC_UTIL = "DCGM_FI_DEV_DEC_UTIL"
    FB_USED = "DCGM_FI_DEV_FB_USED"
    FB_FREE = "DCGM_FI_DEV_FB_FREE"
    GPU_TEMP = "DCGM_FI_DEV_GPU_TEMP"
    MEM_TEMP = "DCGM_FI_DEV_MEM_MAX_TEMP"
    POWER_DRAW = "DCGM_FI_DEV_POWER_USAGE"
    POWER_LIMIT = "DCGM_FI_DEV_POWER_MGMT_LIMIT"
    CLOCK_SM = "DCGM_FI_DEV_SM_CLOCK"
    CLOCK_MEM = "DCGM_FI_DEV_MEM_CLOCK"
    PCIE_TX = "DCGM_FI_DEV_PCIE_TX_THROUGHPUT"
    PCIE_RX = "DCGM_FI_DEV_PCIE_RX_THROUGHPUT"
    NVLINK_TX = "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT"
    NVLINK_RX = "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT"
    XID_ERRORS = "DCGM_FI_DEV_XID_ERRORS"
    ECC_VOLATILE = "DCGM_FI_DEV_ECC_VOLATILE"
    ECC_AGGREGATE = "DCGM_FI_DEV_ECC_AGGREGATE"
    TENSOR_ACTIVITY = "DCGM_FI_DEV_TENSOR_ACTIVITY"
    DRAM_ACTIVITY = "DCGM_FI_DEV_DRAM_ACTIVITY"
    NVLink_CRC_ERRORS = "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERRORS"
    RETIRED_PAGES = "DCGM_FI_DEV_RETIRED_PAGES"
    REMAPPED_ROWS = "DCGM_FI_DEV_REMAPPED_ROWS"


@dataclass
class DcgmSample:
    metric: DcgmMetricType
    gpu_index: int
    gpu_uuid: str
    hostname: str
    value: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class DcgmMetricDiscovery:
    endpoint: str
    available_metrics: list[DcgmMetricType]
    scrape_duration_ms: float
    sample_count: int
    gpu_count: int
    gpu_uuids: list[str]
    last_scrape: str


class DcgmIngestionPipeline:
    def __init__(self, default_endpoint: str = "http://localhost:9400/metrics") -> None:
        self._endpoint = default_endpoint
        self._samples: list[DcgmSample] = []
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._scrape_interval = 15
        self._metric_pattern = re.compile(
            r'^(?P<metric>DCGM_FI_DEV_\w+)'
            r'\{(?P<labels>[^}]+)\}\s+(?P<value>[\d.eE+-]+)',
        )
        self._label_pattern = re.compile(r'(\w+)="([^"]*)"')

    def set_endpoint(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def scrape(self, endpoint: str | None = None) -> DcgmMetricDiscovery:
        url = endpoint or self._endpoint
        started = time.perf_counter()
        samples: list[DcgmSample] = []
        metric_set: set[str] = set()
        gpu_uuids: set[str] = set()
        gpu_indices: set[int] = set()

        try:
            resp = urlopen(url, timeout=10)
            text = resp.read().decode("utf-8")
            for line in text.splitlines():
                m = self._metric_pattern.match(line.strip())
                if not m:
                    continue
                metric_name = m.group("metric")
                labels_str = m.group("labels")
                try:
                    value = float(m.group("value"))
                except ValueError:
                    continue
                labels = dict(self._label_pattern.findall(labels_str))
                gpu_index = int(labels.get("gpu", "-1"))
                gpu_uuid = labels.get("UUID", labels.get("gpu_uuid", ""))
                hostname = labels.get("Hostname", labels.get("hostname", ""))

                try:
                    metric_type = DcgmMetricType(metric_name)
                except ValueError:
                    continue

                metric_set.add(metric_name)
                if gpu_uuid:
                    gpu_uuids.add(gpu_uuid)
                if gpu_index >= 0:
                    gpu_indices.add(gpu_index)
                samples.append(DcgmSample(
                    metric=metric_type, gpu_index=gpu_index,
                    gpu_uuid=gpu_uuid, hostname=hostname, value=value, labels=labels,
                ))

        except Exception as exc:
            logger.warning("DCGM scrape failed: %s", exc)
            return DcgmMetricDiscovery(
                endpoint=url, available_metrics=[], scrape_duration_ms=0,
                sample_count=0, gpu_count=0, gpu_uuids=[], last_scrape="",
            )

        duration_ms = (time.perf_counter() - started) * 1000
        with self._lock:
            self._samples = samples

        return DcgmMetricDiscovery(
            endpoint=url,
            available_metrics=[DcgmMetricType(m) for m in sorted(metric_set)],
            scrape_duration_ms=round(duration_ms, 2),
            sample_count=len(samples),
            gpu_count=len(gpu_indices),
            gpu_uuids=sorted(gpu_uuids),
            last_scrape=datetime.now(timezone.utc).isoformat(),
        )

    def start_polling(self, interval_seconds: int = 15) -> None:
        if self._running:
            return
        self._running = True
        self._scrape_interval = interval_seconds

        def _loop() -> None:
            while self._running:
                try:
                    discovery = self.scrape()
                    logger.debug("DCGM scrape: %d samples from %d GPUs",
                                 discovery.sample_count, discovery.gpu_count)
                except Exception as exc:
                    logger.error("DCGM polling error: %s", exc)
                time.sleep(self._scrape_interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info("DCGM polling started (interval=%ds, endpoint=%s)",
                     interval_seconds, self._endpoint)

    def stop_polling(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def get_latest_samples(self) -> list[DcgmSample]:
        with self._lock:
            return list(self._samples)

    def get_metric(self, metric: DcgmMetricType, gpu_index: int = -1) -> list[DcgmSample]:
        with self._lock:
            result = [s for s in self._samples if s.metric == metric]
            if gpu_index >= 0:
                result = [s for s in result if s.gpu_index == gpu_index]
            return result

    def build_telemetry_samples(self) -> list[dict[str, Any]]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return []

        by_gpu: dict[int, dict[str, Any]] = {}
        for s in samples:
            if s.gpu_index not in by_gpu:
                by_gpu[s.gpu_index] = {"gpu_index": s.gpu_index, "gpu_uuid": s.gpu_uuid}
            key = s.metric.value.lower().replace("dcgm_fi_dev_", "")
            by_gpu[s.gpu_index][key] = s.value

        return list(by_gpu.values())

    @property
    def is_running(self) -> bool:
        return self._running


_default_pipeline = DcgmIngestionPipeline()


def get_dcgm_pipeline() -> DcgmIngestionPipeline:
    return _default_pipeline
