from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class SloSnapshot:
    timestamp: str
    api_availability_pct: float
    check_completion_pct: float
    telemetry_freshness_seconds: float
    state_completeness_pct: float
    audit_durable: bool
    recovery_time_seconds: float | None = None


class SloTracker:
    """Track and report observability SLOs defined in the spec.

    SLO targets (production):
        API availability          ≥ 99.9 %
        Check completion          ≥ 99 % under policy timeout
        Telemetry freshness       ≤ 15 sec for control metrics
        State completeness        ≥ 99 % with explicit missing-data flags
        Audit durability          No acknowledged action without durable audit
        Recovery time             Documented RTO/RPO (automated HA)
    """

    def __init__(self) -> None:
        self._history: list[SloSnapshot] = []

    def record(
        self,
        api_availability_pct: float,
        check_completion_pct: float,
        telemetry_freshness_seconds: float,
        state_completeness_pct: float,
        audit_durable: bool,
        recovery_time_seconds: float | None = None,
    ) -> SloSnapshot:
        snap = SloSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            api_availability_pct=api_availability_pct,
            check_completion_pct=check_completion_pct,
            telemetry_freshness_seconds=telemetry_freshness_seconds,
            state_completeness_pct=state_completeness_pct,
            audit_durable=audit_durable,
            recovery_time_seconds=recovery_time_seconds,
        )
        self._history.append(snap)
        # keep last 1000 entries
        if len(self._history) > 1000:
            self._history = self._history[-500:]
        return snap

    def latest(self) -> SloSnapshot | None:
        return self._history[-1] if self._history else None

    def compliance_summary(self) -> dict[str, Any]:
        latest = self.latest()
        if latest is None:
            return {}
        return {
            "timestamp": latest.timestamp,
            "targets": {
                "api_availability": {
                    "current_pct": latest.api_availability_pct,
                    "target_pct": 99.9,
                    "met": latest.api_availability_pct >= 99.9,
                },
                "check_completion": {
                    "current_pct": latest.check_completion_pct,
                    "target_pct": 99.0,
                    "met": latest.check_completion_pct >= 99.0,
                },
                "telemetry_freshness": {
                    "current_seconds": latest.telemetry_freshness_seconds,
                    "target_seconds": 15.0,
                    "met": latest.telemetry_freshness_seconds <= 15.0,
                },
                "state_completeness": {
                    "current_pct": latest.state_completeness_pct,
                    "target_pct": 99.0,
                    "met": latest.state_completeness_pct >= 99.0,
                },
                "audit_durability": {
                    "current": latest.audit_durable,
                    "required": True,
                    "met": latest.audit_durable is True,
                },
            },
        }


_slo_tracker: SloTracker | None = None


def get_slo_tracker() -> SloTracker:
    global _slo_tracker
    if _slo_tracker is None:
        _slo_tracker = SloTracker()
    return _slo_tracker