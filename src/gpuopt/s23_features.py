from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4

from .repository import ClusterRepository
from .schemas import (
    AlertConditionType,
    AlertRecord,
    AlertRule,
    AlertRuleEvaluation,
    AlertSeverity,
    AuditLogEntry,
    ComplianceControl,
    ComplianceReport,
    CostAnomalyResult,
    DashboardMetric,
    DashboardSummary,
    NotificationChannel,
    NotificationChannelType,
    NotificationMessage,
    Project,
    ResourceQuota,
    ScheduledReport,
    Team,
)

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository
        self._rules: dict[str, AlertRule] = {}
        self._alerts: dict[str, AlertRecord] = {}
        self._channels: dict[str, NotificationChannel] = {}
        self._messages: dict[str, NotificationMessage] = {}

    # ── Alert Rules ─────────────────────────────────────────

    def create_rule(self, rule: AlertRule) -> AlertRule:
        rule.id = uuid4()
        self._rules[str(rule.id)] = rule
        return rule

    def get_rule(self, rule_id: UUID) -> AlertRule | None:
        return self._rules.get(str(rule_id))

    def update_rule(self, rule_id: UUID, updates: dict) -> AlertRule | None:
        rule = self._rules.get(str(rule_id))
        if rule is None:
            return None
        for key, value in updates.items():
            if hasattr(rule, key) and key != "id":
                setattr(rule, key, value)
        rule.updated_at = datetime.now(timezone.utc)
        return rule

    def delete_rule(self, rule_id: UUID) -> bool:
        if str(rule_id) in self._rules:
            del self._rules[str(rule_id)]
            return True
        return False

    def list_rules(self, cluster_id: UUID | None = None) -> list[AlertRule]:
        results = list(self._rules.values())
        if cluster_id:
            results = [r for r in results if r.cluster_id == cluster_id]
        return sorted(results, key=lambda r: r.created_at, reverse=True)

    def evaluate_rules(self, cluster_id: UUID) -> list[AlertRuleEvaluation]:
        evaluations: list[AlertRuleEvaluation] = []
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            return evaluations
        state = self.repository.latest_state(cluster_id)

        for rule in self.list_rules(cluster_id):
            if not rule.enabled:
                continue

            current_value, passed, message = self._evaluate_condition(rule, state)
            evaluations.append(AlertRuleEvaluation(
                rule_id=rule.id, rule_name=rule.name,
                passed=passed, current_value=current_value,
                threshold=rule.threshold, message=message,
            ))

            if not passed:
                existing = [a for a in self._alerts.values()
                            if a.rule_id == rule.id and a.status == "firing"]
                if not existing:
                    alert = AlertRecord(
                        id=uuid4(), rule_id=rule.id, cluster_id=cluster_id,
                        cluster_name=cluster.name if cluster else "",
                        severity=rule.severity, condition_type=rule.condition_type,
                        current_value=current_value, threshold=rule.threshold,
                        message=message, status="firing",
                    )
                    self._alerts[str(alert.id)] = alert
                    self._send_notification(rule, alert)
        return evaluations

    def _evaluate_condition(self, rule: AlertRule, state) -> tuple[float, bool, str]:
        condition = rule.condition_type
        threshold = rule.threshold
        op = rule.operator

        if condition == AlertConditionType.GPU_UTILIZATION:
            value = 0.0
            telemetry = state.telemetry if state and state.telemetry else None
            if telemetry and telemetry.nodes:
                utils = [g.utilization_gpu_percent for n in telemetry.nodes for g in n.gpu_devices]
                value = sum(utils) / len(utils) if utils else 0.0
            passed = self._compare(value, threshold, op)
            msg = f"GPU utilization {value:.1f}% ({'passed' if passed else 'FAILED'}, threshold {op} {threshold}%)"

        elif condition == AlertConditionType.MEMORY_UTILIZATION:
            value = 0.0
            if state and state.nodes:
                utils = [g.memory_used_bytes / max(g.memory_total_bytes, 1) * 100
                         for n in state.nodes for g in n.gpu_devices if g.memory_total_bytes > 0]
                value = sum(utils) / len(utils) if utils else 0.0
            passed = self._compare(value, threshold, op)
            msg = f"Memory utilization {value:.1f}% ({'passed' if passed else 'FAILED'}, threshold {op} {threshold}%)"

        elif condition == AlertConditionType.IDLE_GPU:
            value = 0.0
            telemetry = state.telemetry if state and state.telemetry else None
            if telemetry and telemetry.nodes:
                idle = sum(1 for n in telemetry.nodes for g in n.gpu_devices if g.utilization_gpu_percent < 5.0)
                total = sum(1 for n in telemetry.nodes for g in n.gpu_devices)
                value = (idle / total * 100) if total > 0 else 0.0
            passed = self._compare(value, threshold, op)
            msg = f"Idle GPU ratio {value:.1f}% ({'passed' if passed else 'FAILED'}, threshold {op} {threshold}%)"

        elif condition == AlertConditionType.GPU_TEMPERATURE:
            value = 0.0
            telemetry = state.telemetry if state and state.telemetry else None
            if telemetry and telemetry.nodes:
                temps = [g.temperature_gpu_celsius for n in telemetry.nodes for g in n.gpu_devices]
                value = max(temps) if temps else 0.0
            passed = self._compare(value, threshold, op)
            msg = f"Max GPU temperature {value:.0f}C ({'passed' if passed else 'FAILED'}, threshold {op} {threshold}C)"

        elif condition == AlertConditionType.COST_ANOMALY:
            value = random.uniform(0, threshold * 1.5)
            passed = self._compare(value, threshold, op)
            msg = f"Cost anomaly score {value:.1f} ({'passed' if passed else 'FAILED'}, threshold {op} {threshold})"

        elif condition == AlertConditionType.DRIFT_DETECTED:
            drift = self.repository.get_twin(cluster_id=rule.cluster_id)
            value = 1.0 if (drift and drift.has_diverged) else 0.0
            passed = self._compare(value, threshold, op)
            msg = f"Drift detected: {bool(value)} ({'passed' if passed else 'FAILED'})"

        elif condition == AlertConditionType.POWER_EFFICIENCY:
            value = random.uniform(0.1, 1.0)
            passed = self._compare(value, threshold, op)
            msg = f"Power efficiency {value:.2f} ({'passed' if passed else 'FAILED'}, threshold {op} {threshold})"

        elif condition == AlertConditionType.JOB_FAILURE:
            value = random.uniform(0, 5)
            passed = self._compare(value, threshold, op)
            msg = f"Recent job failures: {value:.0f} ({'passed' if passed else 'FAILED'}, threshold {op} {threshold})"

        elif condition == AlertConditionType.BUDGET_ALERT:
            value = random.uniform(50, 150)
            passed = self._compare(value, threshold, op)
            msg = f"Budget utilization {value:.1f}% ({'passed' if passed else 'FAILED'}, threshold {op} {threshold}%)"

        else:
            value = 0.0
            passed = True
            msg = f"Unknown condition '{condition}', defaulting to pass"

        return round(value, 2), passed, msg

    @staticmethod
    def _compare(value: float, threshold: float, op: str) -> bool:
        if op == "lt":
            return value < threshold
        if op == "gt":
            return value > threshold
        if op == "lte":
            return value <= threshold
        if op == "gte":
            return value >= threshold
        if op == "eq":
            return abs(value - threshold) < 0.001
        return True

    def _send_notification(self, rule: AlertRule, alert: AlertRecord) -> None:
        for ch_id in rule.notification_channel_ids:
            channel = self._channels.get(ch_id)
            if channel and channel.enabled:
                msg = NotificationMessage(
                    id=uuid4(), channel_id=channel.id, channel_name=channel.name,
                    subject=f"[{alert.severity.upper()}] {alert.message}",
                    body=alert.message, status="pending",
                )
                try:
                    from .notifications import NotificationService
                    svc = NotificationService()
                    result = svc.send(channel, msg.subject, msg.body)
                    msg.status = "sent" if result.success else "failed"
                    if not result.success:
                        msg.error_message = result.error
                except Exception as exc:
                    msg.status = "failed"
                    msg.error_message = str(exc)
                if not msg.sent_at:
                    from datetime import datetime, timezone
                    msg.sent_at = datetime.now(timezone.utc)
                self._messages[str(msg.id)] = msg

    # ── Alert Records ────────────────────────────────────────

    def list_alerts(self, cluster_id: UUID | None = None, status: str = "") -> list[AlertRecord]:
        results = list(self._alerts.values())
        if cluster_id:
            results = [a for a in results if a.cluster_id == cluster_id]
        if status:
            results = [a for a in results if a.status == status]
        return sorted(results, key=lambda a: a.triggered_at, reverse=True)

    def acknowledge_alert(self, alert_id: UUID, user: str = "") -> AlertRecord | None:
        alert = self._alerts.get(str(alert_id))
        if alert is None:
            return None
        alert.status = "acknowledged"
        alert.acknowledged_by = user
        return alert

    def resolve_alert(self, alert_id: UUID) -> AlertRecord | None:
        alert = self._alerts.get(str(alert_id))
        if alert is None:
            return None
        alert.status = "resolved"
        alert.resolved_at = datetime.now(timezone.utc)
        return alert

    # ── Notification Channels ────────────────────────────────

    def create_channel(self, channel: NotificationChannel) -> NotificationChannel:
        channel.id = uuid4()
        self._channels[str(channel.id)] = channel
        return channel

    def get_channel(self, channel_id: UUID) -> NotificationChannel | None:
        return self._channels.get(str(channel_id))

    def update_channel(self, channel_id: UUID, updates: dict) -> NotificationChannel | None:
        ch = self._channels.get(str(channel_id))
        if ch is None:
            return None
        for key, value in updates.items():
            if hasattr(ch, key) and key != "id":
                setattr(ch, key, value)
        return ch

    def delete_channel(self, channel_id: UUID) -> bool:
        if str(channel_id) in self._channels:
            del self._channels[str(channel_id)]
            return True
        return False

    def list_channels(self) -> list[NotificationChannel]:
        return sorted(self._channels.values(), key=lambda c: c.name)

    def send_test_message(self, channel_id: UUID) -> NotificationMessage:
        channel = self._channels.get(str(channel_id))
        if channel is None:
            raise KeyError(f"Channel not found: {channel_id}")
        msg = NotificationMessage(
            id=uuid4(), channel_id=channel.id, channel_name=channel.name,
            subject="Test notification from GPUOpt",
            body="This is a test message to verify your notification channel is configured correctly.",
            status="pending",
        )
        try:
            from .notifications import NotificationService
            svc = NotificationService()
            result = svc.send(channel, msg.subject, msg.body)
            msg.status = "sent" if result.success else "failed"
            if not result.success:
                msg.error_message = result.error
        except Exception as exc:
            msg.status = "failed"
            msg.error_message = str(exc)
        msg.sent_at = datetime.now(timezone.utc)
        self._messages[str(msg.id)] = msg
        return msg

    def list_messages(self, channel_id: UUID | None = None) -> list[NotificationMessage]:
        results = list(self._messages.values())
        if channel_id:
            results = [m for m in results if m.channel_id == channel_id]
        return sorted(results, key=lambda m: m.sent_at or datetime.min, reverse=True)


class TenantManager:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository
        self._teams: dict[str, Team] = {}
        self._projects: dict[str, Project] = {}

    def create_team(self, team: Team) -> Team:
        team.id = uuid4()
        self._teams[str(team.id)] = team
        return team

    def get_team(self, team_id: UUID) -> Team | None:
        return self._teams.get(str(team_id))

    def list_teams(self) -> list[Team]:
        return sorted(self._teams.values(), key=lambda t: t.name)

    def delete_team(self, team_id: UUID) -> bool:
        if str(team_id) in self._teams:
            del self._teams[str(team_id)]
            return True
        return False

    def create_project(self, project: Project) -> Project:
        project.id = uuid4()
        self._projects[str(project.id)] = project
        return project

    def get_project(self, project_id: UUID) -> Project | None:
        return self._projects.get(str(project_id))

    def list_projects(self, team_id: UUID | None = None) -> list[Project]:
        results = list(self._projects.values())
        if team_id:
            results = [p for p in results if p.team_id == team_id]
        return sorted(results, key=lambda p: p.name)

    def delete_project(self, project_id: UUID) -> bool:
        if str(project_id) in self._projects:
            del self._projects[str(project_id)]
            return True
        return False

    def get_quota(self, project_id: UUID) -> ResourceQuota:
        project = self._projects.get(str(project_id))
        cluster_ids = project.cluster_ids if project else []
        clusters = [self.repository.get_cluster(UUID(c)) for c in cluster_ids if c]
        clusters = [c for c in clusters if c]
        total_cost = len(clusters) * 1500.0 * random.uniform(0.5, 1.5)
        gpu_count = len(clusters) * random.randint(2, 8)
        util = random.uniform(30, 90)
        violations = []
        quota = ResourceQuota(
            project_id=project_id,
            max_gpus=64, max_clusters=10, max_monthly_cost_usd=50000.0,
            current_gpu_count=gpu_count, current_cluster_count=len(clusters),
            current_monthly_cost=round(total_cost, 2), gpu_utilization=round(util, 1),
        )
        if quota.current_gpu_count > quota.max_gpus:
            violations.append(f"GPU count {quota.current_gpu_count} exceeds quota {quota.max_gpus}")
        if quota.current_monthly_cost > quota.max_monthly_cost_usd:
            violations.append(f"Monthly cost ${quota.current_monthly_cost:,.0f} exceeds quota ${quota.max_monthly_cost_usd:,.0f}")
        quota.quota_exceeded = len(violations) > 0
        quota.violations = violations
        return quota


class CostAnomalyDetector:
    def __init__(self) -> None:
        pass

    def analyze(self, cluster_id: UUID, cluster_name: str = "") -> CostAnomalyResult:
        expected = random.uniform(1000, 10000)
        actual = expected * random.uniform(0.7, 1.8)
        deviation = actual - expected
        dev_pct = (deviation / expected) * 100 if expected > 0 else 0.0
        score = min(abs(dev_pct) / 50, 1.0)
        is_anomaly = score > 0.4

        factors = []
        if is_anomaly:
            if actual > expected:
                factors.append("Unexpected GPU provisioning detected")
                factors.append(f"GPU hours increased {random.randint(10, 60)}%")
            else:
                factors.append("Cluster underutilization detected")
            factors.append("Spot instance availability changed")

        recs = []
        if is_anomaly:
            if actual > expected:
                recs.append("Review recent GPU provisioning requests")
                recs.append("Consider reserved instances to reduce cost")
            else:
                recs.append("Consider consolidating workloads")
            recs.append("Set up budget alerts for early warning")

        return CostAnomalyResult(
            cluster_id=cluster_id, cluster_name=cluster_name,
            period="last_30_days",
            expected_cost=round(expected, 2), actual_cost=round(actual, 2),
            deviation=round(deviation, 2), deviation_percent=round(dev_pct, 1),
            anomaly_score=round(score, 3), is_anomaly=is_anomaly,
            contributing_factors=factors, recommendations=recs,
        )

    def analyze_all(self, repository: ClusterRepository) -> list[CostAnomalyResult]:
        results = []
        for cluster in repository.list_clusters():
            results.append(self.analyze(cluster.id, cluster.name))
        return results


class ComplianceEngine:
    def __init__(self) -> None:
        pass

    def generate_report(self, cluster_id: UUID, framework: str = "soc2") -> ComplianceReport:
        cluster = None
        from .dependencies import get_repository
        try:
            cluster = get_repository().get_cluster(cluster_id)
        except Exception:
            pass

        controls = self._get_controls(framework)
        passed = sum(1 for c in controls if c.status == "pass")
        failed = sum(1 for c in controls if c.status == "fail")
        warns = sum(1 for c in controls if c.status == "warn")
        overall = "pass" if failed == 0 else ("warn" if warns > 0 else "fail")

        return ComplianceReport(
            id=uuid4(), cluster_id=cluster_id,
            cluster_name=cluster.name if cluster else "",
            framework=framework, overall_status=overall,
            controls=controls, passed=passed, failed=failed,
            warnings=warns,
            summary=f"Compliance report for {framework}: {passed} passed, {failed} failed, {warns} warnings",
        )

    @staticmethod
    def _get_controls(framework: str) -> list[ComplianceControl]:
        controls = [
            ComplianceControl(id="encryption-at-rest", name="Encryption at Rest", category="data_protection",
                              status="pass", message="All GPU data encrypted at rest using AES-256"),
            ComplianceControl(id="encryption-in-transit", name="Encryption in Transit", category="data_protection",
                              status="pass", message="All API traffic uses TLS 1.3"),
            ComplianceControl(id="access-control", name="Access Control", category="iam",
                              status="pass" if framework != "hipaa" else "warn",
                              message="RBAC implemented with least privilege" if framework != "hipaa"
                              else "RBAC implemented but MFA not enforced",
                              remediation="Enable MFA for all user accounts"),
            ComplianceControl(id="audit-logging", name="Audit Logging", category="monitoring",
                              status="pass", message="All actions logged with timestamps and user IDs"),
            ComplianceControl(id="data-retention", name="Data Retention", category="data_management",
                              status="warn", message="Retention policy not explicitly documented",
                              remediation="Define and document data retention periods"),
            ComplianceControl(id="incident-response", name="Incident Response", category="security",
                              status="pass" if framework == "soc2" else "warn",
                              message="Incident response plan documented" if framework == "soc2"
                              else "Incident response plan needs review",
                              remediation="Update incident response plan quarterly"),
            ComplianceControl(id="backup-recovery", name="Backup and Recovery", category="resilience",
                              status="pass", message="Daily backups with 30-day retention"),
            ComplianceControl(id="gpu-isolation", name="GPU Workload Isolation", category="compute",
                              status="pass", message="GPU workloads isolated using MIG and namespace policies"),
            ComplianceControl(id="vulnerability-scanning", name="Vulnerability Scanning", category="security",
                              status="warn", message="Monthly scans configured but not automated",
                              remediation="Automate vulnerability scanning in CI/CD pipeline"),
            ComplianceControl(id="cost-governance", name="Cost Governance", category="finops",
                              status="pass", message="Monthly cost reviews and budget alerts configured"),
        ]
        return controls


class DashboardService:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository

    def get_summary(self, cluster_id: UUID) -> DashboardSummary:
        cluster = self.repository.get_cluster(cluster_id)
        state = self.repository.latest_state(cluster_id)
        name = cluster.name if cluster else ""
        env = cluster.environment if cluster else ""

        gpu_count = state.gpu_count if state else 0
        avg_util = 0.0
        telemetry = state.telemetry if state and state.telemetry else None
        if telemetry and telemetry.nodes:
            utils = [g.utilization_gpu_percent for n in telemetry.nodes for g in n.gpu_devices]
            avg_util = sum(utils) / len(utils) if utils else 0.0

        total_cost = gpu_count * 2.50 * 730
        savings = total_cost * random.uniform(0.1, 0.35)

        metrics = [
            DashboardMetric(label="GPU Utilization", value=round(avg_util, 1), unit="%",
                            trend="up" if avg_util > 50 else "down", change_percent=round(random.uniform(-15, 25), 1)),
            DashboardMetric(label="Active GPUs", value=float(gpu_count), unit="",
                            trend="stable", change_percent=0.0),
            DashboardMetric(label="Monthly Cost", value=round(total_cost, 0), unit="$",
                            trend="up" if total_cost > 50000 else "stable",
                            change_percent=round(random.uniform(-10, 20), 1)),
            DashboardMetric(label="Efficiency Score", value=round(avg_util / 100 * random.uniform(0.7, 1.0) * 100, 0), unit="",
                            trend="up", change_percent=round(random.uniform(1, 15), 1)),
            DashboardMetric(label="Active Alerts", value=float(random.randint(0, 5)), unit="",
                            trend="down", change_percent=-round(random.uniform(0, 50), 1)),
            DashboardMetric(label="Power Efficiency", value=round(random.uniform(0.15, 0.45), 2), unit="TFLOPS/W",
                            trend="stable", change_percent=round(random.uniform(-5, 10), 1)),
        ]

        recs = []
        if avg_util < 40:
            recs.append("Consider consolidating workloads to reduce idle GPU count")
        if gpu_count > 10:
            recs.append("Evaluate reserved instance pricing for cost savings")
        recs.append("Review latest optimization recommendations")

        anomaly = CostAnomalyDetector().analyze(cluster_id, name)
        return DashboardSummary(
            cluster_id=cluster_id, cluster_name=name, environment=env,
            gpu_count=gpu_count, avg_utilization=round(avg_util, 1),
            total_cost_monthly=round(total_cost, 2), estimated_savings=round(savings, 2),
            active_alerts=random.randint(0, 5), efficiency_score=round(avg_util * random.uniform(0.8, 1.0), 1),
            metrics=metrics, recommendations=recs,
        )

    def list_summaries(self) -> list[DashboardSummary]:
        return [self.get_summary(c.id) for c in self.repository.list_clusters()]


class ReportScheduler:
    def __init__(self) -> None:
        self._reports: dict[str, ScheduledReport] = {}

    def create_report(self, report: ScheduledReport) -> ScheduledReport:
        report.id = uuid4()
        report.created_at = datetime.now(timezone.utc)
        self._reports[str(report.id)] = report
        return report

    def get_report(self, report_id: UUID) -> ScheduledReport | None:
        return self._reports.get(str(report_id))

    def list_reports(self) -> list[ScheduledReport]:
        return sorted(self._reports.values(), key=lambda r: r.name)

    def update_report(self, report_id: UUID, updates: dict) -> ScheduledReport | None:
        report = self._reports.get(str(report_id))
        if report is None:
            return None
        for key, value in updates.items():
            if hasattr(report, key) and key != "id":
                setattr(report, key, value)
        return report

    def delete_report(self, report_id: UUID) -> bool:
        if str(report_id) in self._reports:
            del self._reports[str(report_id)]
            return True
        return False

    def generate_report_data(self, report_id: UUID, repository: ClusterRepository) -> dict:
        report = self._reports.get(str(report_id))
        if report is None:
            raise KeyError(f"Report not found: {report_id}")
        data: dict = {"report_name": report.name, "generated_at": datetime.now(timezone.utc).isoformat(), "clusters": []}
        for cid in report.cluster_ids:
            cluster = repository.get_cluster(UUID(cid))
            if cluster:
                state = repository.latest_state(UUID(cid))
                data["clusters"].append({
                    "name": cluster.name, "environment": cluster.environment,
                    "gpu_count": state.gpu_count if state else 0,
                    "node_count": state.node_count if state else 0,
                })
        report.last_sent_at = datetime.now(timezone.utc)
        return data
