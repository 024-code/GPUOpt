from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_alert_manager, get_repository
from .repository import ClusterRepository
from .s23_features import AlertManager
from .schemas import (
    AlertConditionType,
    AlertRecord,
    AlertRule,
    AlertRuleEvaluation,
    AlertSeverity,
    NotificationChannel,
    NotificationChannelType,
)

logger = logging.getLogger(__name__)

alert_router = APIRouter(prefix="/api/v1/alerts", tags=["alerting", "notifications"])


# ── Alert Rules ─────────────────────────────────────────────────


@alert_router.post("/rules")
def create_alert_rule(
    rule: AlertRule,
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> AlertRule:
    created = alert_manager.create_rule(rule)
    try:
        repository.save_alert_rule(created)
    except Exception as exc:
        logger.warning("Failed to persist alert rule: %s", exc)
    return created


@alert_router.get("/rules")
def list_alert_rules(
    cluster_id: UUID | None = None,
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> list[AlertRule]:
    return alert_manager.list_rules(cluster_id)


@alert_router.get("/rules/{rule_id}")
def get_alert_rule(
    rule_id: UUID,
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> AlertRule:
    rule = alert_manager.get_rule(rule_id)
    if rule is None:
        raise HTTPException(404, "Alert rule not found")
    return rule


@alert_router.patch("/rules/{rule_id}")
def update_alert_rule(
    rule_id: UUID,
    updates: dict[str, Any],
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> AlertRule:
    rule = alert_manager.update_rule(rule_id, updates)
    if rule is None:
        raise HTTPException(404, "Alert rule not found")
    try:
        repository.save_alert_rule(rule)
    except Exception as exc:
        logger.warning("Failed to persist alert rule update: %s", exc)
    return rule


@alert_router.delete("/rules/{rule_id}")
def delete_alert_rule(
    rule_id: UUID,
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    if not alert_manager.delete_rule(rule_id):
        raise HTTPException(404, "Alert rule not found")
    try:
        repository.delete_alert_rule(rule_id)
    except Exception as exc:
        logger.warning("Failed to delete persisted alert rule: %s", exc)
    return {"status": "deleted"}


@alert_router.post("/rules/{cluster_id}/evaluate")
def evaluate_rules(
    cluster_id: UUID,
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> list[AlertRuleEvaluation]:
    return alert_manager.evaluate_rules(cluster_id)


# ── Alert Records ──────────────────────────────────────────────


@alert_router.get("/records")
def list_alerts(
    cluster_id: UUID | None = None,
    status: str = "",
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> list[AlertRecord]:
    return alert_manager.list_alerts(cluster_id, status)


@alert_router.post("/records/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: UUID,
    user: str = "",
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> AlertRecord:
    alert = alert_manager.acknowledge_alert(alert_id, user)
    if alert is None:
        raise HTTPException(404, "Alert not found")
    try:
        repository.save_alert_record(alert)
    except Exception as exc:
        logger.warning("Failed to persist alert ack: %s", exc)
    return alert


@alert_router.post("/records/{alert_id}/resolve")
def resolve_alert(
    alert_id: UUID,
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> AlertRecord:
    alert = alert_manager.resolve_alert(alert_id)
    if alert is None:
        raise HTTPException(404, "Alert not found")
    try:
        repository.save_alert_record(alert)
    except Exception as exc:
        logger.warning("Failed to persist alert resolve: %s", exc)
    return alert


# ── Notification Channels ──────────────────────────────────────


@alert_router.post("/channels")
def create_notification_channel(
    channel: NotificationChannel,
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> NotificationChannel:
    created = alert_manager.create_channel(channel)
    try:
        repository.save_notification_channel(created)
    except Exception as exc:
        logger.warning("Failed to persist notification channel: %s", exc)
    return created


@alert_router.get("/channels")
def list_channels(
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> list[NotificationChannel]:
    return alert_manager.list_channels()


@alert_router.get("/channels/{channel_id}")
def get_channel(
    channel_id: UUID,
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> NotificationChannel:
    ch = alert_manager.get_channel(channel_id)
    if ch is None:
        raise HTTPException(404, "Notification channel not found")
    return ch


@alert_router.patch("/channels/{channel_id}")
def update_channel(
    channel_id: UUID,
    updates: dict[str, Any],
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> NotificationChannel:
    ch = alert_manager.update_channel(channel_id, updates)
    if ch is None:
        raise HTTPException(404, "Notification channel not found")
    try:
        repository.save_notification_channel(ch)
    except Exception as exc:
        logger.warning("Failed to persist channel update: %s", exc)
    return ch


@alert_router.delete("/channels/{channel_id}")
def delete_channel(
    channel_id: UUID,
    alert_manager: AlertManager = Depends(get_alert_manager),
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    if not alert_manager.delete_channel(channel_id):
        raise HTTPException(404, "Notification channel not found")
    try:
        repository.delete_notification_channel(channel_id)
    except Exception as exc:
        logger.warning("Failed to delete persisted channel: %s", exc)
    return {"status": "deleted"}


@alert_router.post("/channels/{channel_id}/test")
def test_channel(
    channel_id: UUID,
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> dict:
    try:
        msg = alert_manager.send_test_message(channel_id)
    except KeyError:
        raise HTTPException(404, "Notification channel not found")
    return {
        "message_id": str(msg.id),
        "status": msg.status,
        "error": msg.error_message or "",
    }


@alert_router.get("/messages")
def list_messages(
    channel_id: UUID | None = None,
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> list[dict]:
    msgs = alert_manager.list_messages(channel_id)
    return [
        {
            "id": str(m.id),
            "channel_id": str(m.channel_id),
            "channel_name": m.channel_name,
            "subject": m.subject,
            "body": m.body,
            "status": m.status,
            "sent_at": m.sent_at.isoformat() if m.sent_at else None,
            "error_message": m.error_message,
        }
        for m in msgs
    ]
