from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class ExplanationCategory(str, Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    RISK = "risk"
    EFFICIENCY = "efficiency"
    COMPLIANCE = "compliance"
    CAPACITY = "capacity"
    POWER = "power"
    UTILIZATION = "utilization"


@dataclass
class StructuredExplanation:
    explanation_id: str
    recommendation_id: str
    category: ExplanationCategory
    title: str
    summary: str
    root_cause: str
    impact: str
    evidence: list[dict[str, Any]]
    metrics: dict[str, float]
    confidence: float
    severity: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str = ""


@dataclass
class RecommendationExpiry:
    recommendation_id: str
    created_at: str
    ttl_hours: int
    expires_at: str
    is_expired: bool = False
    extended_count: int = 0
    max_extensions: int = 3


@dataclass
class ShadowDeployment:
    shadow_id: str
    recommendation_id: str
    cluster_id: str
    status: str
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    shadow_metrics: dict[str, float] = field(default_factory=dict)
    impact_delta: dict[str, float] = field(default_factory=dict)
    outcome: str = ""
    confidence: float = 0.0
    auto_promote: bool = False
    rollback_on_failure: bool = True


class ExplanationService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._explanations: dict[str, StructuredExplanation] = {}
        self._expiries: dict[str, RecommendationExpiry] = {}
        self._shadows: dict[str, ShadowDeployment] = {}
        self._default_ttl_hours = 720

    def generate_explanation(self, recommendation_id: str, category: ExplanationCategory,
                              title: str, summary: str, root_cause: str, impact: str,
                              evidence: list[dict[str, Any]] | None = None,
                              metrics: dict[str, float] | None = None,
                              confidence: float = 0.8, severity: str = "medium",
                              ttl_hours: int | None = None) -> StructuredExplanation:
        explanation = StructuredExplanation(
            explanation_id=str(uuid4()),
            recommendation_id=recommendation_id,
            category=category, title=title, summary=summary,
            root_cause=root_cause, impact=impact,
            evidence=evidence or [], metrics=metrics or {},
            confidence=confidence, severity=severity,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=ttl_hours or self._default_ttl_hours)).isoformat(),
        )
        with self._lock:
            self._explanations[explanation.explanation_id] = explanation
            self._expiries[recommendation_id] = RecommendationExpiry(
                recommendation_id=recommendation_id,
                created_at=explanation.generated_at,
                ttl_hours=ttl_hours or self._default_ttl_hours,
                expires_at=explanation.expires_at,
            )
        return explanation

    def get_explanation(self, explanation_id: str) -> StructuredExplanation | None:
        with self._lock:
            return self._explanations.get(explanation_id)

    def get_explanation_by_rec(self, recommendation_id: str) -> list[StructuredExplanation]:
        with self._lock:
            return [e for e in self._explanations.values()
                    if e.recommendation_id == recommendation_id]

    def check_expiry(self, recommendation_id: str) -> bool:
        with self._lock:
            exp = self._expiries.get(recommendation_id)
            if exp is None:
                return False
            now = datetime.now(timezone.utc)
            expiry = datetime.fromisoformat(exp.expires_at)
            is_expired = now > expiry
            exp.is_expired = is_expired
            return is_expired

    def extend_expiry(self, recommendation_id: str, hours: int = 168) -> bool:
        with self._lock:
            exp = self._expiries.get(recommendation_id)
            if exp is None or exp.extended_count >= exp.max_extensions:
                return False
            exp.expires_at = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
            exp.extended_count += 1
            exp.is_expired = False
            return True

    def create_shadow(self, recommendation_id: str, cluster_id: str,
                       baseline_metrics: dict[str, float] | None = None,
                       auto_promote: bool = False,
                       rollback_on_failure: bool = True) -> ShadowDeployment:
        shadow = ShadowDeployment(
            shadow_id=str(uuid4()),
            recommendation_id=recommendation_id,
            cluster_id=cluster_id,
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
            baseline_metrics=baseline_metrics or {},
            auto_promote=auto_promote,
            rollback_on_failure=rollback_on_failure,
        )
        with self._lock:
            self._shadows[shadow.shadow_id] = shadow
        return shadow

    def start_shadow(self, shadow_id: str) -> ShadowDeployment | None:
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if shadow is None or shadow.status != "pending":
                return None
            shadow.status = "running"
            shadow.started_at = datetime.now(timezone.utc).isoformat()
            return shadow

    def complete_shadow(self, shadow_id: str, shadow_metrics: dict[str, float],
                         outcome: str = "success", confidence: float = 0.0) -> ShadowDeployment | None:
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if shadow is None:
                return None
            shadow.status = "completed"
            shadow.completed_at = datetime.now(timezone.utc).isoformat()
            shadow.shadow_metrics = shadow_metrics
            shadow.outcome = outcome
            shadow.confidence = confidence or 0.8
            if shadow.baseline_metrics:
                shadow.impact_delta = {
                    k: shadow_metrics.get(k, 0) - v
                    for k, v in shadow.baseline_metrics.items()
                }
            return shadow

    def fail_shadow(self, shadow_id: str, reason: str = "") -> ShadowDeployment | None:
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if shadow is None:
                return None
            shadow.status = "failed"
            shadow.completed_at = datetime.now(timezone.utc).isoformat()
            shadow.outcome = reason or "failed"
            return shadow

    def promote_shadow(self, shadow_id: str) -> dict[str, Any]:
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if shadow is None or shadow.status != "completed":
                return {"success": False, "error": "Shadow not completed"}
            if shadow.impact_delta and any(v < 0 for v in shadow.impact_delta.values()):
                logger.warning("Shadow %s has negative impact, not auto-promoting", shadow_id)
                return {"success": False, "error": "Negative impact detected"}
            return {"success": True, "shadow_id": shadow_id, "action": "promote",
                    "impact_delta": shadow.impact_delta}

    def list_shadows(self, cluster_id: str | None = None,
                     status: str | None = None) -> list[ShadowDeployment]:
        with self._lock:
            shadows = list(self._shadows.values())
            if cluster_id:
                shadows = [s for s in shadows if s.cluster_id == cluster_id]
            if status:
                shadows = [s for s in shadows if s.status == status]
            return shadows

    def get_shadow(self, shadow_id: str) -> ShadowDeployment | None:
        with self._lock:
            return self._shadows.get(shadow_id)

    def list_expired_recommendations(self) -> list[dict[str, Any]]:
        with self._lock:
            now = datetime.now(timezone.utc)
            expired = []
            for rec_id, exp in self._expiries.items():
                if datetime.fromisoformat(exp.expires_at) < now:
                    expired.append({
                        "recommendation_id": rec_id,
                        "created_at": exp.created_at,
                        "expires_at": exp.expires_at,
                        "extended_count": exp.extended_count,
                    })
            return expired


_explanation_service = ExplanationService()


def get_explanation_service() -> ExplanationService:
    return _explanation_service
