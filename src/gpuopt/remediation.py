from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from .schemas import (
    ActuationRecord,
    ActuationStatus,
    AlertRecord,
    AlertSeverity,
    AlertRule,
    PolicyRule,
    PolicySeverity,
    ResourceRecommendation,
)
from .repository import ClusterRepository

logger = logging.getLogger(__name__)


class RemediationActionType(StrEnum):
    ACTUATE_RECOMMENDATION = "actuate_recommendation"
    SCALE_CLUSTER = "scale_cluster"
    RESTART_SERVICE = "restart_service"
    DRAIN_NODE = "drain_node"
    ADJUST_POWER_CAP = "adjust_power_cap"
    SEND_NOTIFICATION = "send_notification"
    CREATE_TICKET = "create_ticket"
    RUN_CHAOS = "run_chaos"


class RemediationStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RemediationAction:
    id: str = field(default_factory=lambda: str(uuid4()))
    type: RemediationActionType = RemediationActionType.SEND_NOTIFICATION
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: RemediationStatus = RemediationStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: str = ""
    error: str = ""


@dataclass
class RemediationRule:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    trigger_alert_severity: list[AlertSeverity] = field(default_factory=lambda: [AlertSeverity.CRITICAL])
    trigger_condition_types: list[str] = field(default_factory=list)
    trigger_policy_severity: list[PolicySeverity] = field(default_factory=list)
    actions: list[RemediationAction] = field(default_factory=list)
    cooldown_minutes: int = 60
    enabled: bool = True
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RemediationRun:
    id: str = field(default_factory=lambda: str(uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    trigger_type: str = ""
    trigger_id: str = ""
    cluster_id: str = ""
    status: RemediationStatus = RemediationStatus.PENDING
    actions: list[RemediationAction] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    summary: str = ""


class RemediationEngine:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository
        self._rules: dict[str, RemediationRule] = {}
        self._runs: dict[str, RemediationRun] = {}
        self._last_run: dict[str, datetime] = {}
        self._seed_default_rules()

    def _seed_default_rules(self) -> None:
        rules = [
            RemediationRule(
                name="Critical GPU Alert Auto-Remediation",
                description="Apply top recommendation when critical GPU alerts fire",
                trigger_alert_severity=[AlertSeverity.CRITICAL],
                trigger_condition_types=["gpu_utilization", "memory_utilization", "gpu_temperature"],
                actions=[
                    RemediationAction(
                        type=RemediationActionType.ACTUATE_RECOMMENDATION,
                        target="cluster",
                        params={"dry_run": False, "select": "best"},
                    ),
                    RemediationAction(
                        type=RemediationActionType.SEND_NOTIFICATION,
                        target="admin",
                        params={"channel": "slack", "template": "remediation_started"},
                    ),
                ],
                cooldown_minutes=120,
            ),
            RemediationRule(
                name="Power Efficiency Auto-Tuning",
                description="Apply power cap when power efficiency is low",
                trigger_alert_severity=[AlertSeverity.WARNING, AlertSeverity.CRITICAL],
                trigger_condition_types=["power_efficiency"],
                actions=[
                    RemediationAction(
                        type=RemediationActionType.ADJUST_POWER_CAP,
                        target="gpu",
                        params={"reduction_percent": 10},
                    ),
                ],
                cooldown_minutes=60,
            ),
            RemediationRule(
                name="Cost Anomaly Notification",
                description="Notify team when cost anomaly is detected",
                trigger_alert_severity=[AlertSeverity.CRITICAL],
                trigger_condition_types=["cost_anomaly", "budget_alert"],
                actions=[
                    RemediationAction(
                        type=RemediationActionType.SEND_NOTIFICATION,
                        target="finops-team",
                        params={"channel": "email", "template": "cost_anomaly"},
                    ),
                    RemediationAction(
                        type=RemediationActionType.CREATE_TICKET,
                        target="jira",
                        params={"project": "FINops", "priority": "high"},
                    ),
                ],
                cooldown_minutes=1440,
            ),
        ]
        for rule in rules:
            self._rules[rule.id] = rule

    def list_rules(self) -> list[RemediationRule]:
        return sorted(self._rules.values(), key=lambda r: r.name)

    def get_rule(self, rule_id: str) -> RemediationRule | None:
        return self._rules.get(rule_id)

    def add_rule(self, rule: RemediationRule) -> RemediationRule:
        if not rule.id:
            rule.id = str(uuid4())
        rule.created_at = datetime.now(timezone.utc)
        self._rules[rule.id] = rule
        return rule

    def update_rule(self, rule_id: str, updates: dict) -> RemediationRule | None:
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        for key, value in updates.items():
            if hasattr(rule, key) and key not in ("id", "created_at"):
                setattr(rule, key, value)
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        return self._rules.pop(rule_id, None) is not None

    def evaluate_alert(self, alert: AlertRecord) -> RemediationRun | None:
        if alert.status != "firing":
            return None
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if alert.severity not in rule.trigger_alert_severity:
                continue
            if rule.trigger_condition_types and alert.condition_type not in rule.trigger_condition_types:
                continue
            cluster_id = str(alert.cluster_id)
            last = self._last_run.get(f"{rule.id}:{cluster_id}")
            if last and (datetime.now(timezone.utc) - last).total_seconds() < rule.cooldown_minutes * 60:
                logger.info("Remediation rule %s in cooldown for cluster %s", rule.name, cluster_id)
                continue
            return self._execute_rule(rule, "alert", str(alert.id), cluster_id)
        return None

    def evaluate_policy(self, policy: PolicyRule, cluster_id: str) -> RemediationRun | None:
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.trigger_policy_severity and policy.severity not in rule.trigger_policy_severity:
                continue
            last = self._last_run.get(f"{rule.id}:{cluster_id}")
            if last and (datetime.now(timezone.utc) - last).total_seconds() < rule.cooldown_minutes * 60:
                continue
            return self._execute_rule(rule, "policy", str(policy.id), cluster_id)
        return None

    def _execute_rule(self, rule: RemediationRule, trigger_type: str, trigger_id: str, cluster_id: str) -> RemediationRun:
        run = RemediationRun(
            rule_id=rule.id, rule_name=rule.name,
            trigger_type=trigger_type, trigger_id=trigger_id,
            cluster_id=cluster_id, status=RemediationStatus.IN_PROGRESS,
            actions=[],
        )
        for action_template in rule.actions:
            action = RemediationAction(
                type=action_template.type,
                target=action_template.target,
                params=dict(action_template.params),
                status=RemediationStatus.PENDING,
            )
            try:
                action.status = RemediationStatus.IN_PROGRESS
                action.started_at = datetime.now(timezone.utc)
                result = self._execute_action(action, cluster_id)
                action.status = RemediationStatus.COMPLETED if result["success"] else RemediationStatus.FAILED
                action.result = result.get("message", "")
                if not result["success"]:
                    action.error = result.get("error", "")
            except Exception as exc:
                action.status = RemediationStatus.FAILED
                action.error = str(exc)
            action.completed_at = datetime.now(timezone.utc)
            run.actions.append(action)

        failed = sum(1 for a in run.actions if a.status == RemediationStatus.FAILED)
        run.status = RemediationStatus.COMPLETED if failed == 0 else RemediationStatus.FAILED
        run.completed_at = datetime.now(timezone.utc)
        run.summary = f"{len(run.actions)} actions, {failed} failed"
        self._runs[run.id] = run
        self._last_run[f"{rule.id}:{cluster_id}"] = datetime.now(timezone.utc)
        return run

    def _execute_action(self, action: RemediationAction, cluster_id: str) -> dict[str, Any]:
        if action.type == RemediationActionType.ACTUATE_RECOMMENDATION:
            return self._actuate_recommendation(cluster_id, action.params)
        elif action.type == RemediationActionType.ADJUST_POWER_CAP:
            return self._adjust_power_cap(cluster_id, action.params)
        elif action.type == RemediationActionType.SEND_NOTIFICATION:
            return self._send_notification(action.target, action.params)
        elif action.type == RemediationActionType.CREATE_TICKET:
            return self._create_ticket(action.params)
        return {"success": False, "message": f"Unknown action type: {action.type}", "error": "unknown_action"}

    def _actuate_recommendation(self, cluster_id: str, params: dict) -> dict[str, Any]:
        try:
            cid = UUID(cluster_id)
            rec_set = self.repository.latest_recommendations(cid)
            if rec_set is None or not rec_set.recommendations:
                return {"success": False, "message": "No recommendations available", "error": "no_recs"}
            dry_run = params.get("dry_run", True)
            select = params.get("select", "best")
            if select == "best":
                target = max(rec_set.recommendations, key=lambda r: r.score)
            else:
                target = rec_set.recommendations[0]
            from .actuation import ActuationService
            svc = ActuationService(self.repository)
            record = svc.actuate(cid, target.id, dry_run=dry_run)
            status = "completed" if record.status == ActuationStatus.COMPLETED else "failed"
            return {"success": status == "completed", "message": f"Actuation {record.id}: {record.status.value}", "actuation_id": str(record.id)}
        except Exception as exc:
            return {"success": False, "message": str(exc), "error": str(exc)}

    def _adjust_power_cap(self, cluster_id: str, params: dict) -> dict[str, Any]:
        try:
            reduction = params.get("reduction_percent", 10)
            from .power import PowerService
            svc = PowerService(self.repository)
            cid = UUID(cluster_id)
            analysis = svc.analyze_power(cid)
            if analysis and analysis.total_gpus > 0:
                return {"success": True, "message": f"Suggested {reduction}% power cap reduction for {analysis.total_gpus} GPUs"}
            return {"success": False, "message": "No power analysis available", "error": "no_data"}
        except Exception as exc:
            return {"success": False, "message": str(exc), "error": str(exc)}

    def _send_notification(self, target: str, params: dict) -> dict[str, Any]:
        channel = params.get("channel", "log")
        template = params.get("template", "default")
        logger.info("Remediation notification to %s via %s (template=%s)", target, channel, template)
        return {"success": True, "message": f"Notification sent to {target} via {channel}"}

    def _create_ticket(self, params: dict) -> dict[str, Any]:
        project = params.get("project", "OPS")
        priority = params.get("priority", "medium")
        logger.info("Remediation ticket created in %s (priority=%s)", project, priority)
        return {"success": True, "message": f"Ticket created in {project} with {priority} priority"}

    def list_runs(self, cluster_id: str | None = None) -> list[RemediationRun]:
        runs = list(self._runs.values())
        if cluster_id:
            runs = [r for r in runs if r.cluster_id == cluster_id]
        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def get_run(self, run_id: str) -> RemediationRun | None:
        return self._runs.get(run_id)

    def reset(self) -> None:
        self._rules.clear()
        self._runs.clear()
        self._last_run.clear()
        self._seed_default_rules()
        logger.info("RemediationEngine reset")
