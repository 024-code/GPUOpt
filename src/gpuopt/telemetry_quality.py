from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .schemas import (
    OnboardingPhase,
    TelemetryFallbackRecord,
    TelemetryQualityScore,
    TelemetrySnapshot,
    TelemetrySourceContract,
    TelemetrySourceStatus,
)

logger = logging.getLogger(__name__)

_FALLBACK_CACHE: dict[str, dict] = {}
_DEFAULT_VALUES: dict[str, Any] = {
    "gpu_count": 0,
    "gpu_utilization_percent": 0.0,
    "memory_used_gb": 0.0,
    "temperature_celsius": 50.0,
    "queue_depth": 0,
    "requests_per_second": 0.0,
}


# ── 1. Data Quality Scoring ───────────────────────────────────

class TelemetryQualityScorer:
    def score_snapshot(self, snapshot: TelemetrySnapshot) -> TelemetryQualityScore:
        completeness = self._score_completeness(snapshot)
        freshness = self._score_freshness(snapshot)
        consistency = self._score_consistency(snapshot)
        accuracy = self._score_accuracy(snapshot)
        overall = round(completeness * 0.3 + freshness * 0.3 + consistency * 0.2 + accuracy * 0.2, 3)
        issues = self._collect_issues(completeness, freshness, consistency, accuracy, snapshot)
        return TelemetryQualityScore(
            source="telemetry_snapshot",
            completeness=round(completeness, 3),
            freshness=round(freshness, 3),
            consistency=round(consistency, 3),
            accuracy=round(accuracy, 3),
            overall=overall,
            issues=issues,
        )

    def _score_completeness(self, snapshot: TelemetrySnapshot) -> float:
        total = 0
        present = 0
        checks = [
            ("gpu_snapshot", bool(snapshot.gpu_snapshot)),
            ("model_services", len(snapshot.model_services) > 0),
            ("fabric", len(snapshot.fabric) > 0),
            ("queues", len(snapshot.queues) > 0),
            ("jobs", len(snapshot.jobs) > 0),
        ]
        for _name, ok in checks:
            total += 1
            if ok:
                present += 1
        if snapshot.gpu_snapshot:
            gpu_fields = ["total_gpus", "devices", "total_memory_mb", "used_memory_mb"]
            for f in gpu_fields:
                total += 1
                if f in snapshot.gpu_snapshot:
                    present += 1
        return present / max(total, 1)

    def _score_freshness(self, snapshot: TelemetrySnapshot) -> float:
        try:
            collected = datetime.fromisoformat(snapshot.collected_at)
            age = (datetime.now(timezone.utc) - collected).total_seconds()
            if age < 15:
                return 1.0
            if age < 60:
                return max(0.3, 1.0 - age / 60 * 0.7)
            return max(0.0, 1.0 - age / 300)
        except Exception:
            return 0.0

    def _score_consistency(self, snapshot: TelemetrySnapshot) -> float:
        issues = 0
        checks = 0
        gpu = snapshot.gpu_snapshot
        if gpu:
            checks += 1
            devices = gpu.get("devices", [])
            for d in devices:
                mem_total = d.get("memory_total_mb", 0)
                mem_used = d.get("memory_used_mb", 0)
                if mem_used > mem_total:
                    issues += 1
                util = d.get("utilization_percent", 0)
                if util < 0 or util > 100:
                    issues += 1
                temp = d.get("temperature_celsius", 0)
                if temp < 0 or temp > 110:
                    issues += 1
                checks += 1
        for q in snapshot.queues:
            checks += 2
            if q.pending_jobs > q.queue_depth:
                issues += 1
            if q.completion_rate_per_minute > q.submission_rate_per_minute + 5:
                issues += 1
        for j in snapshot.jobs:
            checks += 1
            if j.memory_required_gb > 0 and j.gpu_utilization_avg > 100:
                issues += 1
        return max(0.0, 1.0 - issues / max(checks, 1))

    def _score_accuracy(self, snapshot: TelemetrySnapshot) -> float:
        score = 1.0
        gpu = snapshot.gpu_snapshot
        if gpu:
            devices = gpu.get("devices", [])
            for d in devices:
                if d.get("power_draw_watts", 0) > 1000:
                    score -= 0.05
                if d.get("temperature_celsius", 0) > 105:
                    score -= 0.05
        return max(0.1, score)

    def _collect_issues(self, comp: float, fresh: float, cons: float, acc: float,
                        snapshot: TelemetrySnapshot) -> list[str]:
        issues = []
        if comp < 0.7:
            issues.append(f"Low completeness ({comp:.0%})")
        if fresh < 0.3:
            issues.append(f"Stale data (freshness {fresh:.0%})")
        if cons < 0.7:
            issues.append(f"Inconsistent data ({cons:.0%})")
        if acc < 0.7:
            issues.append(f"Low accuracy ({acc:.0%})")
        if not snapshot.gpu_snapshot:
            issues.append("GPU snapshot missing")
        if not snapshot.queues:
            issues.append("Queue telemetry missing")
        return issues


# ── 2. Stale-Data Fallback ────────────────────────────────────

class TelemetryFallbackHandler:
    def __init__(self, max_cache_age: float = 300.0) -> None:
        self._max_cache_age = max_cache_age
        self._lock = threading.Lock()

    def store(self, source: str, data: dict) -> None:
        with self._lock:
            _FALLBACK_CACHE[source] = {
                "data": data,
                "timestamp": time.time(),
            }
            if len(_FALLBACK_CACHE) > 100:
                oldest = min(_FALLBACK_CACHE.keys(),
                             key=lambda k: _FALLBACK_CACHE[k]["timestamp"])
                del _FALLBACK_CACHE[oldest]

    def resolve(self, source: str, live_data: dict | None,
                required_fields: list[str] | None = None) -> tuple[dict, TelemetryFallbackRecord]:
        fields = required_fields or list(_DEFAULT_VALUES.keys())
        record = TelemetryFallbackRecord(source=source)
        now = time.time()

        with self._lock:
            cached_entry = _FALLBACK_CACHE.get(source)

        if live_data is not None:
            record.primary_available = True
            record.data_age_seconds = 0.0
            missing = [f for f in fields if f not in live_data]
            if not missing:
                record.using_fallback = False
                self.store(source, live_data)
                return live_data, record
            record.fallback_reason = f"Missing fields: {missing}"
            record.fallback_tier = "degraded"

            if cached_entry and (now - cached_entry["timestamp"]) < self._max_cache_age:
                record.cache_available = True
                record.data_age_seconds = now - cached_entry["timestamp"]
                record.using_fallback = True
                merged = dict(cached_entry["data"])
                merged.update(live_data)
                for f in fields:
                    if f not in merged:
                        merged[f] = _DEFAULT_VALUES.get(f, 0)
                self.store(source, merged)
                return merged, record

            for f in missing:
                if f in _DEFAULT_VALUES:
                    live_data[f] = _DEFAULT_VALUES[f]
            self.store(source, live_data)
            return live_data, record

        record.primary_available = False
        record.fallback_reason = "Primary data unavailable"
        record.fallback_tier = "cached"

        if cached_entry and (now - cached_entry["timestamp"]) < self._max_cache_age:
            record.cache_available = True
            record.data_age_seconds = now - cached_entry["timestamp"]
            record.using_fallback = True
            merged = dict(cached_entry["data"])
            for f in fields:
                if f not in merged:
                    merged[f] = _DEFAULT_VALUES.get(f, 0)
            return merged, record

        record.using_fallback = True
        record.fallback_tier = "default"
        record.data_age_seconds = 999.0
        record.fallback_reason = "No cached data available"
        defaults = dict(_DEFAULT_VALUES)
        return defaults, record

    def get_cache_health(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                "cache_size": len(_FALLBACK_CACHE),
                "sources": list(_FALLBACK_CACHE.keys()),
                "oldest_entry_seconds": round(
                    min((now - c["timestamp"]) for c in _FALLBACK_CACHE.values()), 1
                ) if _FALLBACK_CACHE else 0,
            }


# ── 3. Phased Onboarding ──────────────────────────────────────

class OnboardingManager:
    PHASES = [
        {"name": "discovery", "checks": ["endpoint_reachable", "auth_configured"]},
        {"name": "contract", "checks": ["schema_validated", "required_fields_present", "interval_agreed"]},
        {"name": "sandbox", "checks": ["data_received", "quality_above_0_5", "latency_acceptable"]},
        {"name": "bronze", "checks": ["quality_above_0_7", "stale_fallback_configured", "alerts_configured"]},
        {"name": "silver", "checks": ["quality_above_0_85", "consistent_24h", "error_rate_below_0_01"]},
        {"name": "gold", "checks": ["quality_above_0_95", "sla_achieved_7d", "redundancy_configured"]},
    ]

    def __init__(self) -> None:
        self._sources: dict[str, TelemetrySourceStatus] = {}
        self._contracts: dict[str, TelemetrySourceContract] = {}
        self._phases: dict[str, list[OnboardingPhase]] = {}

    def register_source(self, name: str, source_type: str,
                        required_fields: list[str] | None = None,
                        critical: bool = False) -> TelemetrySourceStatus:
        sid = f"src-{len(self._sources) + 1}"
        contract = TelemetrySourceContract(
            source_id=sid, name=name, source_type=source_type,
            required_fields=required_fields or [],
            critical=critical,
        )
        self._contracts[sid] = contract
        phases = []
        for i, pdef in enumerate(self.PHASES):
            phase = OnboardingPhase(
                name=pdef["name"],
                description=f"Phase {i + 1}: {pdef['name'].title()}",
                status="in_progress" if i == 0 else "pending",
                checks_passed=0,
                checks_total=len(pdef["checks"]),
                started_at=datetime.now(timezone.utc).isoformat() if i == 0 else "",
            )
            phases.append(phase)
        self._phases[sid] = phases
        status = TelemetrySourceStatus(
            source_id=sid, name=name, source_type=source_type,
            phase="discovery", tier="onboarding",
            contract_valid=True,
        )
        self._sources[sid] = status
        logger.info("Source %s (%s) registered for phased onboarding", name, sid)
        return status

    def advance_phase(self, source_id: str, checks_passed: int | None = None) -> dict:
        phases = self._phases.get(source_id)
        status = self._sources.get(source_id)
        if not phases or not status:
            return {"error": "Source not found"}
        for i, phase in enumerate(phases):
            if phase.status == "in_progress":
                passed = checks_passed if checks_passed is not None else phase.checks_total
                phase.checks_passed = min(passed, phase.checks_total)
                phase.readiness_score = round(phase.checks_passed / max(phase.checks_total, 1), 3)
                if phase.checks_passed >= phase.checks_total:
                    phase.status = "completed"
                    phase.completed_at = datetime.now(timezone.utc).isoformat()
                    status.phase = self.PHASES[i]["name"]
                    status.tier = self.PHASES[i]["name"]
                    if i + 1 < len(phases):
                        phases[i + 1].status = "in_progress"
                        phases[i + 1].started_at = datetime.now(timezone.utc).isoformat()
                    return {
                        "source_id": source_id,
                        "phase_completed": phase.name,
                        "next_phase": phases[i + 1].name if i + 1 < len(phases) else None,
                        "tier": status.tier,
                    }
                return {
                    "source_id": source_id,
                    "phase": phase.name,
                    "checks_passed": phase.checks_passed,
                    "checks_total": phase.checks_total,
                    "readiness": phase.readiness_score,
                }
        return {"source_id": source_id, "message": "All phases completed", "tier": status.tier}

    def get_status(self, source_id: str) -> TelemetrySourceStatus | None:
        return self._sources.get(source_id)

    def list_sources(self) -> list[dict]:
        return [
            {
                "id": sid,
                "name": s.name,
                "type": s.source_type,
                "phase": s.phase,
                "tier": s.tier,
                "issues": s.issues,
            }
            for sid, s in self._sources.items()
        ]

    def get_phases(self, source_id: str) -> list[OnboardingPhase] | None:
        return self._phases.get(source_id)

    def update_quality(self, source_id: str, quality: TelemetryQualityScore) -> None:
        status = self._sources.get(source_id)
        if status:
            status.quality = quality
            if quality.overall < 0.5:
                status.issues.append(f"Quality below threshold: {quality.overall:.0%}")
            if quality.overall >= 0.95:
                self.advance_phase(source_id)


# ── Unified R01 Service ───────────────────────────────────────

class TelemetryQualityService:
    def __init__(self) -> None:
        self.scorer = TelemetryQualityScorer()
        self.fallback = TelemetryFallbackHandler()
        self.onboarding = OnboardingManager()

    def process_snapshot(self, snapshot: TelemetrySnapshot) -> dict:
        quality = self.scorer.score_snapshot(snapshot)
        self.fallback.store("telemetry_snapshot", snapshot.model_dump(mode="json"))
        return {
            "quality": quality.model_dump(mode="json"),
            "needs_fallback": quality.overall < 0.5,
            "action": "accept" if quality.overall >= 0.5 else "warn",
        }

    def get_data(self, source: str, live_data: dict | None = None,
                 required_fields: list[str] | None = None) -> dict:
        data, record = self.fallback.resolve(source, live_data, required_fields)
        return {
            "data": data,
            "fallback": record.model_dump(mode="json"),
        }

    def health(self) -> dict:
        cache = self.fallback.get_cache_health()
        sources = self.onboarding.list_sources()
        return {
            "status": "healthy",
            "cache": cache,
            "sources_onboarded": len(sources),
            "sources": sources,
        }
