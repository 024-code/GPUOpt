from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RemediationAction(str, Enum):
    CORDON = "cordon"
    DRAIN = "drain"
    RESTART = "restart"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBOOT = "reboot"


@dataclass
class RemediationResult:
    node_id: str
    action: RemediationAction
    status: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float = 0.0


class AutoHealer:
    def __init__(self) -> None:
        self.remediation_history: list[RemediationResult] = []
        self._active_remediations: dict[str, RemediationAction] = {}
        self._rng = random.Random()

    def check_node_health(self, telemetry: dict) -> dict:
        issues: list[str] = []
        risk_score = 0.0

        temp = telemetry.get("temperature", 0)
        if temp > 85:
            issues.append(f"Critical temperature: {temp}°C")
            risk_score += 0.4
        elif temp > 75:
            issues.append(f"High temperature: {temp}°C")
            risk_score += 0.2

        ecc = telemetry.get("ecc_errors", 0)
        if ecc > 20:
            issues.append(f"Critical ECC errors: {ecc}")
            risk_score += 0.4
        elif ecc > 10:
            issues.append(f"Elevated ECC errors: {ecc}")
            risk_score += 0.2

        mem = telemetry.get("memory_utilization", 0)
        if mem > 95:
            issues.append(f"Critical memory pressure: {mem}%")
            risk_score += 0.3
        elif mem > 85:
            issues.append(f"High memory pressure: {mem}%")
            risk_score += 0.15

        util = telemetry.get("gpu_utilization", 0)
        if util > 95:
            issues.append(f"GPU saturation: {util}%")
            risk_score += 0.2

        xid = telemetry.get("xid_errors", 0)
        if xid > 10:
            issues.append(f"Critical XID errors: {xid}")
            risk_score += 0.5
        elif xid > 5:
            issues.append(f"XID errors detected: {xid}")
            risk_score += 0.25

        health_status = "healthy"
        if risk_score >= 0.8:
            health_status = "critical"
        elif risk_score >= 0.4:
            health_status = "degraded"
        elif risk_score >= 0.2:
            health_status = "warning"

        return {
            "health_status": health_status,
            "risk_score": round(risk_score, 2),
            "issues": issues,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def suggest_remediation(self, telemetry: dict, node_id: str = "unknown") -> RemediationResult | None:
        health = self.check_node_health(telemetry)
        risk = health["risk_score"]

        if risk < 0.4:
            return None

        issues_str = ", ".join(health["issues"]) if health["issues"] else f"risk_score={risk}"

        if risk >= 0.8:
            action = RemediationAction.CORDON
            message = f"Cordoning node {node_id}: {issues_str}"
        elif risk >= 0.6:
            action = RemediationAction.DRAIN
            message = f"Draining node {node_id}: {issues_str}"
        else:
            action = RemediationAction.RESTART
            message = f"Restarting GPU stack on node {node_id}: {issues_str}"

        result = RemediationResult(
            node_id=node_id,
            action=action,
            status="proposed",
            message=message,
        )
        return result

    def execute_remediation(self, telemetry: dict, node_id: str = "unknown") -> RemediationResult:
        suggested = self.suggest_remediation(telemetry, node_id)
        if suggested is None:
            return RemediationResult(
                node_id=node_id,
                action=RemediationAction.SCALE_DOWN,
                status="skipped",
                message=f"No remediation needed for node {node_id}",
            )

        duration = random.uniform(0.5, 5.0)
        suggested.status = "executed"
        suggested.duration_seconds = round(duration, 2)
        self.remediation_history.append(suggested)
        self._active_remediations[node_id] = suggested.action
        logger.info("Executed %s on %s (%.2fs)", suggested.action.value, node_id, duration)
        return suggested

    def get_history(self, limit: int = 50) -> list[dict]:
        return [
            {
                "node_id": r.node_id,
                "action": r.action.value,
                "status": r.status,
                "message": r.message,
                "timestamp": r.timestamp,
                "duration_seconds": r.duration_seconds,
            }
            for r in self.remediation_history[-limit:]
        ]

    def get_active_remediations(self) -> dict[str, str]:
        return {node: action.value for node, action in self._active_remediations.items()}

    def clear_history(self) -> None:
        self.remediation_history.clear()
        self._active_remediations.clear()
