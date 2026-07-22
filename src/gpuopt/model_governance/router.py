from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from .governance import ModelGovernor, get_governor
from .models import (
    ApprovalRequest,
    ApprovalStatus,
    DriftReport,
    GovernanceConfig,
    ModelActionClass,
    ModelStatus,
    ModelVersion,
    ShadowEvaluation,
)

logger = logging.getLogger(__name__)

governance_router = APIRouter(prefix="/api/v1/governance", tags=["governance"])


def _gov() -> ModelGovernor:
    return get_governor()


# ── Model Registry ────────────────────────────────────────────

@governance_router.post("/models/register")
def register_model(
    model_name: str,
    version: str,
    action_class: ModelActionClass,
    owner: str = "",
    features: list[str] = Query(default=[]),
    training_metrics: dict[str, float] = {},
    _gov: ModelGovernor = Depends(_gov),
) -> ModelVersion:
    return _gov.register_model(model_name, version, action_class, owner, features or None, training_metrics or None)


@governance_router.get("/models")
def list_models(
    action_class: ModelActionClass | None = None,
    status: ModelStatus | None = None,
    limit: int = Query(100, le=500),
    _gov: ModelGovernor = Depends(_gov),
) -> list[ModelVersion]:
    return _gov.registry.list(action_class=action_class, status=status, limit=limit)


@governance_router.get("/models/{version_id}")
def get_model(version_id: UUID, _gov: ModelGovernor = Depends(_gov)) -> ModelVersion | None:
    return _gov.registry.get(version_id)


@governance_router.patch("/models/{version_id}/status")
def update_model_status(
    version_id: UUID,
    status: ModelStatus,
    _gov: ModelGovernor = Depends(_gov),
) -> ModelVersion | None:
    return _gov.registry.update_status(version_id, status)


@governance_router.get("/metadata/{action_class}")
def get_metadata(action_class: ModelActionClass, _gov: ModelGovernor = Depends(_gov)) -> object:
    return _gov.registry.get_metadata(action_class.value.title(), action_class)


# ── Champion / Challenger ─────────────────────────────────────

@governance_router.post("/champion-challenger/start")
def start_evaluation(
    champion_id: UUID,
    challenger_id: UUID,
    _gov: ModelGovernor = Depends(_gov),
) -> ShadowEvaluation:
    champion = _gov.registry.get(champion_id)
    challenger = _gov.registry.get(challenger_id)
    if champion is None or challenger is None:
        raise ValueError("Champion or challenger not found")
    return _gov.champion_challenger.start_evaluation(champion, challenger)


@governance_router.post("/champion-challenger/record")
def record_result(
    evaluation_id: UUID,
    champion_score: float,
    challenger_score: float,
    _gov: ModelGovernor = Depends(_gov),
) -> ShadowEvaluation:
    return _gov.champion_challenger.record_result(evaluation_id, champion_score, challenger_score)


@governance_router.post("/champion-challenger/complete/{evaluation_id}")
def complete_evaluation(evaluation_id: UUID, _gov: ModelGovernor = Depends(_gov)) -> ShadowEvaluation:
    return _gov.champion_challenger.complete_evaluation(evaluation_id)


@governance_router.get("/champion-challenger/evaluations")
def list_evaluations(
    action_class: ModelActionClass | None = None,
    _gov: ModelGovernor = Depends(_gov),
) -> list[ShadowEvaluation]:
    return _gov.champion_challenger.list_evaluations(action_class)


@governance_router.get("/champion-challenger/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: UUID, _gov: ModelGovernor = Depends(_gov)) -> ShadowEvaluation | None:
    return _gov.champion_challenger.get_evaluation(evaluation_id)


# ── Drift ─────────────────────────────────────────────────────

@governance_router.get("/drift/reports")
def list_drift_reports(
    version_id: UUID | None = None,
    limit: int = Query(100, le=500),
    _gov: ModelGovernor = Depends(_gov),
) -> list[DriftReport]:
    return _gov.drift_monitor.list_reports(version_id=version_id, limit=limit)


@governance_router.post("/drift/check/{version_id}")
def check_drift(version_id: UUID, _gov: ModelGovernor = Depends(_gov)) -> list[DriftReport]:
    version = _gov.registry.get(version_id)
    if version is None:
        raise ValueError("Model version not found")
    return _gov.drift_monitor.check_all(version)


@governance_router.post("/drift/record-features/{version_id}")
def record_features(version_id: UUID, features: dict[str, float], _gov: ModelGovernor = Depends(_gov)) -> object:
    version = _gov.registry.get(version_id)
    if version is None:
        raise ValueError("Model version not found")
    _gov.drift_monitor.record_features(version, features)
    return {"status": "ok", "feature_count": len(features)}


# ── Fallback ──────────────────────────────────────────────────

@governance_router.get("/fallback/status/{action_class}")
def fallback_status(action_class: ModelActionClass, _gov: ModelGovernor = Depends(_gov)) -> object:
    return {
        "action_class": action_class.value,
        "active": _gov.fallback.is_fallback_active(action_class),
    }


@governance_router.post("/fallback/activate")
def activate_fallback(action_class: ModelActionClass, reason: str = "", _gov: ModelGovernor = Depends(_gov)) -> object:
    _gov.fallback.activate_fallback(action_class, reason)
    return {"status": "ok", "action_class": action_class.value, "active": True}


@governance_router.post("/fallback/deactivate")
def deactivate_fallback(action_class: ModelActionClass, _gov: ModelGovernor = Depends(_gov)) -> object:
    _gov.fallback.deactivate_fallback(action_class)
    return {"status": "ok", "active": False}


# ── Approval ──────────────────────────────────────────────────

@governance_router.get("/approval/requests")
def list_approval_requests(
    status: ApprovalStatus | None = None,
    action_class: ModelActionClass | None = None,
    limit: int = Query(100, le=500),
    _gov: ModelGovernor = Depends(_gov),
) -> list[ApprovalRequest]:
    return _gov.approval.list_requests(status=status, action_class=action_class, limit=limit)


@governance_router.post("/approval/approve/{request_id}")
def approve_request(
    request_id: UUID,
    reviewer: str,
    notes: str = "",
    certification_days: int = 90,
    _gov: ModelGovernor = Depends(_gov),
) -> ApprovalRequest | None:
    return _gov.approval.approve(request_id, reviewer, notes, certification_days)


@governance_router.post("/approval/reject/{request_id}")
def reject_request(
    request_id: UUID,
    reviewer: str,
    notes: str = "",
    _gov: ModelGovernor = Depends(_gov),
) -> ApprovalRequest | None:
    return _gov.approval.reject(request_id, reviewer, notes)


# ── Governance lifecycle ──────────────────────────────────────

@governance_router.post("/periodic-check")
def periodic_check(_gov: ModelGovernor = Depends(_gov)) -> object:
    return _gov.run_periodic_checks()


@governance_router.get("/config")
def get_config(_gov: ModelGovernor = Depends(_gov)) -> object:
    return _gov._config.model_dump(mode="json")
