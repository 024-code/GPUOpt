from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .dcgm_ingestion import DcgmIngestionPipeline, DcgmMetricType, DcgmSample

logger = logging.getLogger(__name__)


class DataQuality(str, Enum):
    GOOD = "good"
    STALE = "stale"
    MISSING = "missing"
    SUSPICIOUS = "suspicious"
    ERROR = "error"
    NOT_AVAILABLE = "not_available"


@dataclass
class GpuQualityFlags:
    gpu_index: int
    gpu_uuid: str
    data_quality: DataQuality
    has_recent_data: bool
    seconds_since_last_sample: float
    xid_errors_recent: int
    ecc_errors_aggregate: int
    retired_pages: int
    remapped_rows: int
    temperature_warning: bool
    power_warning: bool
    utilization_anomaly: bool
    clock_throttling: bool
    pcie_link_degraded: bool
    nvlink_crc_errors: int
    quality_score: float
    warnings: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ClusterQualityReport:
    cluster_id: str
    total_gpus: int
    healthy_gpus: int
    warning_gpus: int
    critical_gpus: int
    overall_quality: DataQuality
    gpu_flags: list[GpuQualityFlags]
    quality_score: float
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DcgmQualityAnalyzer:
    def __init__(self, pipeline: DcgmIngestionPipeline | None = None) -> None:
        self._pipeline = pipeline
        self._stale_threshold_seconds = 60
        self._temp_warning_celsius = 85.0
        self._power_warning_watts = 400.0
        self._xid_error_threshold = 1
        self._ecc_error_threshold = 100

    def analyze_gpu(self, samples: dict[str, Any], gpu_index: int, gpu_uuid: str,
                    now: float) -> GpuQualityFlags:
        warnings: list[str] = []
        quality_score = 100.0
        stale = 0.0
        xid_errors = 0
        ecc_agg = 0
        retired = 0
        remapped = 0
        temp_warn = False
        power_warn = False
        util_anomaly = False
        clock_throttle = False
        pcie_degraded = False
        nvlink_crc = 0

        has_recent = "DCGM_FI_DEV_GPU_UTIL" in samples
        temp = float(samples.get("DCGM_FI_DEV_GPU_TEMP", 0))
        power = float(samples.get("DCGM_FI_DEV_POWER_USAGE", 0))
        util = float(samples.get("DCGM_FI_DEV_GPU_UTIL", -1))
        mem_util = float(samples.get("DCGM_FI_DEV_MEM_COPY_UTIL", -1))
        clock_sm = float(samples.get("DCGM_FI_DEV_SM_CLOCK", 0))
        clock_mem = float(samples.get("DCGM_FI_DEV_MEM_CLOCK", 0))
        fb_used = float(samples.get("DCGM_FI_DEV_FB_USED", 0))
        fb_free = float(samples.get("DCGM_FI_DEV_FB_FREE", 0))
        pcie_tx = float(samples.get("DCGM_FI_DEV_PCIE_TX_THROUGHPUT", -1))
        nvlink_crc = int(float(samples.get("DCGM_FI_DEV_NVLINK_CRC_FLIT_ERRORS", 0)))

        xid_errors = int(float(samples.get("DCGM_FI_DEV_XID_ERRORS", 0)))
        ecc_agg = int(float(samples.get("DCGM_FI_DEV_ECC_AGGREGATE", 0)))
        retired = int(float(samples.get("DCGM_FI_DEV_RETIRED_PAGES", 0)))
        remapped = int(float(samples.get("DCGM_FI_DEV_REMAPPED_ROWS", 0)))

        if stale > self._stale_threshold_seconds:
            warnings.append(f"Data stale ({stale:.0f}s old)")
            quality_score -= 20
        if temp > self._temp_warning_celsius:
            temp_warn = True
            warnings.append(f"High temperature: {temp:.0f}C")
            quality_score -= 15
        if power > self._power_warning_watts:
            power_warn = True
            warnings.append(f"High power draw: {power:.0f}W")
            quality_score -= 10
        if xid_errors >= self._xid_error_threshold:
            warnings.append(f"XID errors detected: {xid_errors}")
            quality_score -= 25
        if ecc_agg >= self._ecc_error_threshold:
            warnings.append(f"High ECC errors: {ecc_agg}")
            quality_score -= 15
        if retired > 0:
            warnings.append(f"Retired pages: {retired}")
            quality_score -= 10
        if remapped > 0:
            warnings.append(f"Remapped rows: {remapped}")
            quality_score -= 5
        if nvlink_crc > 0:
            warnings.append(f"NVLink CRC errors: {nvlink_crc}")
            quality_score -= 10

        if util >= 0 and mem_util >= 0:
            if util < 1 and mem_util < 1 and (fb_used > 0 or power > 0):
                util_anomaly = True
                warnings.append("Utilization anomaly: GPU active but no reported utilization")
                quality_score -= 10

        if clock_sm <= 0 and clock_mem <= 0 and util > 0:
            clock_throttle = True
            warnings.append("Clock throttling suspected")
            quality_score -= 5

        if pcie_tx == 0 and power > 100:
            pcie_degraded = True
            warnings.append("PCIE link may be degraded")
            quality_score -= 5

        quality_score = max(0, min(100, quality_score))
        if quality_score >= 80:
            data_quality = DataQuality.GOOD
        elif quality_score >= 50:
            data_quality = DataQuality.SUSPICIOUS
        elif quality_score >= 20:
            data_quality = DataQuality.STALE
        else:
            data_quality = DataQuality.ERROR

        return GpuQualityFlags(
            gpu_index=gpu_index, gpu_uuid=gpu_uuid,
            data_quality=data_quality, has_recent_data=has_recent,
            seconds_since_last_sample=stale,
            xid_errors_recent=xid_errors, ecc_errors_aggregate=ecc_agg,
            retired_pages=retired, remapped_rows=remapped,
            temperature_warning=temp_warn, power_warning=power_warn,
            utilization_anomaly=util_anomaly, clock_throttling=clock_throttle,
            pcie_link_degraded=pcie_degraded, nvlink_crc_errors=nvlink_crc,
            quality_score=round(quality_score, 1), warnings=warnings,
        )

    def analyze_cluster(self, cluster_id: str,
                        pipeline: DcgmIngestionPipeline | None = None) -> ClusterQualityReport:
        pipe = pipeline or self._pipeline
        samples = pipe.get_latest_samples() if pipe else []
        now = datetime.now(timezone.utc).timestamp()

        gpu_data: dict[int, dict[str, str | float]] = {}
        gpu_uuids: dict[int, str] = {}
        for s in samples:
            if s.gpu_index not in gpu_data:
                gpu_data[s.gpu_index] = {}
                gpu_uuids[s.gpu_index] = s.gpu_uuid
            gpu_data[s.gpu_index][s.metric.value] = s.value

        flags = [
            self.analyze_gpu(gpu_data[idx], idx, gpu_uuids.get(idx, ""), now)
            for idx in sorted(gpu_data.keys())
        ]

        if not flags:
            return ClusterQualityReport(
                cluster_id=cluster_id, total_gpus=0, healthy_gpus=0,
                warning_gpus=0, critical_gpus=0,
                overall_quality=DataQuality.NOT_AVAILABLE,
                gpu_flags=[], quality_score=0.0,
            )

        healthy = sum(1 for f in flags if f.data_quality == DataQuality.GOOD)
        warning = sum(1 for f in flags if f.data_quality in (DataQuality.SUSPICIOUS, DataQuality.STALE))
        critical = sum(1 for f in flags if f.data_quality == DataQuality.ERROR)
        avg_quality = sum(f.quality_score for f in flags) / len(flags)

        if critical > 0:
            overall = DataQuality.ERROR
        elif warning > healthy:
            overall = DataQuality.SUSPICIOUS
        else:
            overall = DataQuality.GOOD

        return ClusterQualityReport(
            cluster_id=cluster_id, total_gpus=len(flags),
            healthy_gpus=healthy, warning_gpus=warning,
            critical_gpus=critical, overall_quality=overall,
            gpu_flags=flags, quality_score=round(avg_quality, 1),
        )
