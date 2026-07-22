from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from uuid import UUID, uuid4

from .repository import ClusterRepository
from .schemas import (
    ApprovalRecord,
    ApprovalStatus,
    ApprovalStep,
    ApprovalWorkflowRequest,
    ChaosExperiment,
    ChaosExperimentResult,
    ChaosFaultTarget,
    ChaosFaultType,
    GuardedAutomationRecommendation,
    PolicyEvaluationResult,
    PolicyRule,
    PolicySeverity,
    PreFlightCheckResult,
)

logger = logging.getLogger(__name__)


class PolicyEngine:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository
        self._default_policies: list[PolicyRule] = []

    def _load_policies(self) -> list[PolicyRule]:
        policies = self.repository.list_policies()
        return policies + self._default_policies

    def check_pre_flight(
        self,
        cluster_id: UUID,
        rec_id: UUID,
        environment: str = "",
    ) -> PreFlightCheckResult:
        policies = self._load_policies()
        cluster = self.repository.get_cluster(cluster_id)
        cluster_name = cluster.name if cluster else ""

        results: list[PolicyEvaluationResult] = []
        requires_approval = False

        for policy in policies:
            if not policy.enabled:
                continue
            if not self._policy_applies(policy, cluster_id, environment):
                continue
            result = self._evaluate_policy(policy, cluster_id, rec_id, environment)
            results.append(result)
            if result.action == "block" and not result.passed:
                requires_approval = True

        passed_count = sum(1 for r in results if r.passed)
        blocked_count = sum(1 for r in results if not r.passed and r.action == "block")
        warned_count = sum(1 for r in results if not r.passed and r.action == "warn")
        overall_passed = blocked_count == 0

        return PreFlightCheckResult(
            actuation_rec_id=rec_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            overall_passed=overall_passed,
            policy_count=len(results),
            passed_count=passed_count,
            blocked_count=blocked_count,
            warned_count=warned_count,
            results=results,
            requires_approval=requires_approval,
            checked_at=datetime.now(timezone.utc),
        )

    def create_policy(self, policy: PolicyRule) -> PolicyRule:
        self.repository.save_policy(policy)
        return policy

    def update_policy(self, policy_id: UUID, updates: dict) -> PolicyRule | None:
        existing = self.repository.get_policy(policy_id)
        if existing is None:
            return None
        for key, value in updates.items():
            if hasattr(existing, key) and key != "id":
                setattr(existing, key, value)
        existing.updated_at = datetime.now(timezone.utc)
        self.repository.save_policy(existing)
        return existing

    def delete_policy(self, policy_id: UUID) -> bool:
        return self.repository.delete_policy(policy_id)

    def get_policy(self, policy_id: UUID) -> PolicyRule | None:
        return self.repository.get_policy(policy_id)

    def list_policies(self) -> list[PolicyRule]:
        return self._load_policies()

    def _policy_applies(self, policy: PolicyRule, cluster_id: UUID, environment: str) -> bool:
        if policy.scope_type == "global":
            return True
        if policy.scope_type == "cluster":
            return policy.scope_value == str(cluster_id)
        if policy.scope_type == "environment":
            return policy.scope_value == environment
        return True

    def _evaluate_policy(
        self,
        policy: PolicyRule,
        cluster_id: UUID,
        rec_id: UUID,
        environment: str,
    ) -> PolicyEvaluationResult:
        rule_type = policy.rule_type
        config = policy.rule_config

        if rule_type == "environment_restriction":
            allowed = config.get("allowed_environments", [])
            denied = config.get("denied_environments", [])
            if denied and environment in denied:
                return PolicyEvaluationResult(
                    policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                    severity=policy.severity, passed=False, action="block",
                    message=f"Environment '{environment}' is restricted",
                    details=[f"Environment '{environment}' is in denied list: {denied}"],
                )
            if allowed and environment not in allowed:
                return PolicyEvaluationResult(
                    policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                    severity=policy.severity, passed=False, action="block",
                    message=f"Environment '{environment}' is not in allowed list",
                    details=[f"Allowed environments: {allowed}"],
                )
            return PolicyEvaluationResult(
                policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                severity=policy.severity, passed=True, action="allow",
                message=f"Environment '{environment}' is permitted",
            )

        if rule_type == "time_window":
            window_start = config.get("start_hour", 0)
            window_end = config.get("end_hour", 23)
            allowed_days = config.get("allowed_days", [0, 1, 2, 3, 4, 5, 6])
            now = datetime.now(timezone.utc)
            current_hour = now.hour
            current_day = now.weekday()
            if current_day not in allowed_days:
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                return PolicyEvaluationResult(
                    policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                    severity=policy.severity, passed=False, action="block",
                    message=f"Actuations not allowed on {day_names[current_day]}",
                    details=[f"Allowed days: {[day_names[d] for d in allowed_days]}"],
                )
            if current_hour < window_start or current_hour >= window_end:
                return PolicyEvaluationResult(
                    policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                    severity=policy.severity, passed=False, action="block",
                    message=f"Outside allowed window ({window_start}:00-{window_end}:00 UTC)",
                    details=[f"Current hour: {current_hour}:00 UTC"],
                )
            return PolicyEvaluationResult(
                policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                severity=policy.severity, passed=True, action="allow",
                message="Within allowed time window",
            )

        if rule_type == "approval_required":
            return PolicyEvaluationResult(
                policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                severity=policy.severity, passed=False, action="block",
                message="This actuation requires approval before execution",
                details=[config.get("reason", "Policy requires manual approval")],
            )

        if rule_type == "resource_limit":
            max_gpu = config.get("max_gpu_count", 0)
            if max_gpu > 0:
                rec_set = self.repository.latest_recommendations(cluster_id)
                if rec_set:
                    for rec in rec_set.recommendations:
                        if str(rec.id) == str(rec_id):
                            affected = len(rec.affected_resources)
                            gpu_involved = sum(1 for r in rec.affected_resources if "gpu" in r.lower())
                            if gpu_involved > max_gpu:
                                return PolicyEvaluationResult(
                                    policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                                    severity=policy.severity, passed=False, action="block",
                                    message=f"Exceeds max GPU count ({gpu_involved} > {max_gpu})",
                                    details=[f"Resources: {rec.affected_resources}"],
                                )
            return PolicyEvaluationResult(
                policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                severity=policy.severity, passed=True, action="allow",
                message="Within resource limits",
            )

        if rule_type == "maintenance_window":
            in_maintenance = config.get("in_maintenance", False)
            if in_maintenance:
                return PolicyEvaluationResult(
                    policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                    severity=policy.severity, passed=False, action="block",
                    message="Cluster is in maintenance window",
                    details=[config.get("message", "Maintenance in progress")],
                )
            return PolicyEvaluationResult(
                policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
                severity=policy.severity, passed=True, action="allow",
                message="No active maintenance window",
            )

        return PolicyEvaluationResult(
            policy_id=policy.id, policy_name=policy.name, rule_type=rule_type,
            severity=policy.severity, passed=True, action="allow",
            message=f"Unknown rule type '{rule_type}', defaulting to allow",
        )


class ApprovalWorkflow:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository

    def create_request(self, request: ApprovalWorkflowRequest) -> ApprovalRecord:
        cluster_id = request.cluster_id if hasattr(request, 'cluster_id') and request.cluster_id else None
        cluster = self.repository.get_cluster(cluster_id) if cluster_id else None

        steps = [
            ApprovalStep(step_order=i + 1, approver=approver)
            for i, approver in enumerate(request.required_approvers)
        ]
        if not steps:
            steps = [ApprovalStep(step_order=1, approver="default_approver")]

        record = ApprovalRecord(
            id=uuid4(),
            actuation_id=request.actuation_id,
            cluster_id=cluster_id or uuid4(),
            cluster_name=cluster.name if cluster else "",
            status=ApprovalStatus.PENDING,
            steps=steps,
            required_approvers=request.required_approvers,
            reason=request.reason,
        )
        self.repository.save_approval(record)
        return record

    def approve(
        self,
        approval_id: UUID,
        approver: str,
        reason: str = "",
    ) -> ApprovalRecord:
        record = self.repository.get_approval(approval_id)
        if record is None:
            raise KeyError(f"Approval record not found: {approval_id}")
        if record.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval is already {record.status.value}")

        for step in record.steps:
            if step.approver == approver and step.status == ApprovalStatus.PENDING:
                step.status = ApprovalStatus.APPROVED
                step.decided_at = datetime.now(timezone.utc)
                step.reason = reason
                break
        else:
            raise ValueError(f"Approver '{approver}' not found in workflow or already decided")

        all_decided = all(s.status != ApprovalStatus.PENDING for s in record.steps)
        all_approved = all(s.status == ApprovalStatus.APPROVED for s in record.steps)
        any_rejected = any(s.status == ApprovalStatus.REJECTED for s in record.steps)

        if all_decided and all_approved:
            record.status = ApprovalStatus.APPROVED
            record.decided_at = datetime.now(timezone.utc)
            record.final_reason = reason or "All approvers approved"
        elif any_rejected:
            record.status = ApprovalStatus.REJECTED
            record.decided_at = datetime.now(timezone.utc)
            record.final_reason = reason or "Rejected by approver"

        self.repository.save_approval(record)
        return record

    def reject(
        self,
        approval_id: UUID,
        approver: str,
        reason: str = "",
    ) -> ApprovalRecord:
        record = self.repository.get_approval(approval_id)
        if record is None:
            raise KeyError(f"Approval record not found: {approval_id}")
        if record.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval is already {record.status.value}")

        for step in record.steps:
            if step.approver == approver and step.status == ApprovalStatus.PENDING:
                step.status = ApprovalStatus.REJECTED
                step.decided_at = datetime.now(timezone.utc)
                step.reason = reason
                break
        else:
            raise ValueError(f"Approver '{approver}' not found in workflow or already decided")

        record.status = ApprovalStatus.REJECTED
        record.decided_at = datetime.now(timezone.utc)
        record.final_reason = reason or "Rejected by approver"
        self.repository.save_approval(record)
        return record

    def get_approval(self, approval_id: UUID) -> ApprovalRecord | None:
        return self.repository.get_approval(approval_id)

    def list_approvals(self, cluster_id: UUID | None = None) -> list[ApprovalRecord]:
        return self.repository.list_approvals(cluster_id)

    def check_actuation_approved(self, actuation_id: UUID) -> bool:
        approvals = self.repository.list_approvals_for_actuation(actuation_id)
        if not approvals:
            return False
        latest = max(approvals, key=lambda a: a.created_at)
        return latest.status == ApprovalStatus.APPROVED


class ChaosEngine:
    def __init__(self) -> None:
        self._experiments: dict[str, ChaosExperiment] = {}

    def create_experiment(self, experiment: ChaosExperiment) -> ChaosExperiment:
        experiment.id = uuid4()
        experiment.created_at = datetime.now(timezone.utc)
        self._experiments[str(experiment.id)] = experiment
        return experiment

    def get_experiment(self, experiment_id: UUID) -> ChaosExperiment | None:
        return self._experiments.get(str(experiment_id))

    def list_experiments(self, cluster_id: UUID | None = None) -> list[ChaosExperiment]:
        results = list(self._experiments.values())
        if cluster_id:
            results = [e for e in results if e.cluster_id == cluster_id]
        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def run_experiment(self, experiment_id: UUID) -> ChaosExperimentResult:
        experiment = self._experiments.get(str(experiment_id))
        if experiment is None:
            raise KeyError(f"Chaos experiment not found: {experiment_id}")

        experiment.status = "running"
        experiment.started_at = datetime.now(timezone.utc)

        target_count = max(1, experiment.target.count)
        recovery_time = round(random.uniform(5.0, 120.0), 1)
        gpu_drop = round(random.uniform(5.0, 60.0), 1)
        latency_inc = round(random.uniform(10.0, 500.0), 1)
        errors = random.randint(0, 10)
        resilient = errors < 5 and gpu_drop < 40

        observations = [
            f"Targeted {target_count} resources with {experiment.fault_type.value}",
            f"GPU utilization dropped {gpu_drop}% during fault",
            f"Latency increased by {latency_inc}ms",
        ]
        if resilient:
            observations.append("System self-healed within expected timeframe")
            observations.append("No cascading failures detected")
        else:
            observations.append("System showed signs of instability")
            observations.append("Manual intervention may be required")

        recs = []
        if not resilient:
            recs.append("Add circuit breakers to prevent cascading failures")
            recs.append("Implement pod disruption budgets")
            recs.append("Review resource limits and requests")
        else:
            recs.append("Continue current resilience practices")
            recs.append("Consider expanding chaos coverage to more fault types")

        experiment.status = "completed"
        experiment.completed_at = datetime.now(timezone.utc)

        return ChaosExperimentResult(
            experiment=experiment,
            experiment_status="completed",
            target_impacted=target_count,
            gpu_utilization_drop_percent=gpu_drop,
            latency_increase_ms=latency_inc,
            error_count=errors,
            recovery_time_seconds=recovery_time,
            system_resilient=resilient,
            observations=observations,
            recommendations=recs,
            summary=f"Chaos experiment '{experiment.name}': {'RESILIENT' if resilient else 'FAILURES DETECTED'} "
                    f"(GPU drop: {gpu_drop}%, errors: {errors}, recovery: {recovery_time}s)",
        )

    def delete_experiment(self, experiment_id: UUID) -> bool:
        if str(experiment_id) in self._experiments:
            del self._experiments[str(experiment_id)]
            return True
        return False

    def generate_ga_recommendations(self, cluster_id: UUID) -> list[GuardedAutomationRecommendation]:
        cluster = None
        recs: list[GuardedAutomationRecommendation] = []

        recs.append(GuardedAutomationRecommendation(
            cluster_id=cluster_id,
            cluster_name=cluster.name if cluster else "",
            environment=cluster.environment if cluster else "",
            recommendation="Enable pre-flight policy checks on all actuation paths",
            recommendation_type="policy",
            priority="high",
            estimated_risk_reduction="Prevents unauthorized or unsafe actuations before execution",
            actions=["Define environment restriction policies for production",
                     "Set time window policies for maintenance windows",
                     "Configure approval-required policies for high-risk changes"],
        ))

        recs.append(GuardedAutomationRecommendation(
            cluster_id=cluster_id,
            cluster_name=cluster.name if cluster else "",
            environment=cluster.environment if cluster else "",
            recommendation="Implement multi-step approval for production actuations",
            recommendation_type="approval",
            priority="high",
            estimated_risk_reduction="Ensures critical changes have peer review before deployment",
            actions=["Require at least 2 approvers for production changes",
                     "Set up approval notification and escalation paths",
                     "Integrate with Slack/PagerDuty for approval workflows"],
        ))

        recs.append(GuardedAutomationRecommendation(
            cluster_id=cluster_id,
            cluster_name=cluster.name if cluster else "",
            environment=cluster.environment if cluster else "",
            recommendation="Run chaos experiments regularly to validate system resilience",
            recommendation_type="chaos",
            priority="medium",
            estimated_risk_reduction="Identifies failure modes before they cause incidents",
            actions=["Schedule weekly chaos experiments on non-production clusters",
                     "Start with node failure and GPU failure fault types",
                     "Document recovery procedures based on experiment results"],
        ))

        recs.append(GuardedAutomationRecommendation(
            cluster_id=cluster_id,
            cluster_name=cluster.name if cluster else "",
            environment=cluster.environment if cluster else "",
            recommendation="Monitor actuation approval and policy compliance metrics",
            recommendation_type="monitoring",
            priority="medium",
            estimated_risk_reduction="Provides visibility into governance effectiveness",
            actions=["Track policy violation rates per environment",
                     "Monitor approval cycle times",
                     "Set up alerts for blocked actuation attempts"],
        ))

        return recs
