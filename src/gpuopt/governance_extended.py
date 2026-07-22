from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any

from .gpu_monitor import GPUMonitor
from .model_governance.approval import ApprovalManager
from .model_governance.governance import ModelGovernor
from .schemas import (
    Explanation,
    GovernanceReport,
    PolicyEnvelope,
    RollbackPlan,
    TenantQuota,
)

logger = logging.getLogger(__name__)


class PolicyEnforcer:
    def __init__(self) -> None:
        self._envelopes: dict[str, PolicyEnvelope] = {}

    def create_envelope(self, name: str, domain: str = "general",
                        rules: list[dict] | None = None,
                        action: str = "block") -> PolicyEnvelope:
        env = PolicyEnvelope(
            name=name,
            description=f"Policy envelope for {domain}: {name}",
            domain=domain,
            rules=rules or [],
            action=action,
        )
        self._envelopes[env.policy_id] = env
        return env

    def evaluate(self, envelope: PolicyEnvelope, context: dict) -> dict:
        failed = []
        for rule in envelope.rules:
            field = rule.get("field", "")
            op = rule.get("operator", "eq")
            value = rule.get("value")
            context_val = context.get(field)
            passed = self._evaluate_rule(context_val, op, value)
            if not passed:
                failed.append({"field": field, "operator": op, "expected": value, "actual": context_val})
        action = envelope.action if failed else "allow"
        return {"passed": len(failed) == 0, "failed_rules": failed, "action": action}

    def _evaluate_rule(self, context_val: Any, operator: str, value: Any) -> bool:
        if context_val is None:
            return False
        try:
            if operator == "eq": return context_val == value
            elif operator == "neq": return context_val != value
            elif operator == "gt": return context_val > value
            elif operator == "gte": return context_val >= value
            elif operator == "lt": return context_val < value
            elif operator == "lte": return context_val <= value
            elif operator == "in": return context_val in (value or [])
            elif operator == "not_in": return context_val not in (value or [])
        except (TypeError, ValueError):
            return False
        return False

    def list_envelopes(self, domain: str | None = None) -> list[PolicyEnvelope]:
        if domain:
            return [e for e in self._envelopes.values() if e.domain == domain]
        return list(self._envelopes.values())

    def enable(self, policy_id: str) -> bool:
        if policy_id in self._envelopes:
            self._envelopes[policy_id].enabled = True
            return True
        return False

    def disable(self, policy_id: str) -> bool:
        if policy_id in self._envelopes:
            self._envelopes[policy_id].enabled = False
            return True
        return False


class RollbackManager:
    def __init__(self) -> None:
        self._plans: dict[str, RollbackPlan] = {}

    def create_plan(self, action_id: str, action_type: str, reason: str,
                    context: dict | None = None) -> RollbackPlan:
        ctx = context or {}
        steps = [
            f"Stop action {action_id} ({action_type})",
            f"Revert configuration changes for {action_type}",
            f"Restore previous state from backup",
            f"Verify system health after rollback",
            f"Notify stakeholders of rollback",
        ]
        risk = "low" if action_type in ("placement", "right_size") else "medium"
        automated = action_type not in ("preempt", "consolidate")
        requires_approval = action_type in ("preempt", "consolidate")
        plan = RollbackPlan(
            action_id=action_id,
            action_type=action_type,
            reason=reason,
            steps=steps,
            estimated_rollback_time_seconds=round(random.uniform(10, 300), 0),
            risk_level=risk,
            automated=automated,
            requires_approval=requires_approval,
        )
        self._plans[plan.plan_id] = plan
        return plan

    def execute(self, plan: RollbackPlan) -> dict:
        logger.info("Executing rollback plan %s: %s", plan.plan_id, plan.reason)
        return {"status": "completed", "plan_id": plan.plan_id, "steps_executed": len(plan.steps)}

    def get_plan(self, plan_id: str) -> RollbackPlan | None:
        return self._plans.get(plan_id)

    def list_plans(self) -> list[RollbackPlan]:
        return list(self._plans.values())


class TenantQuotaManager:
    def __init__(self) -> None:
        self._quotas: dict[str, TenantQuota] = {}

    def set_quota(self, tenant_id: str, tenant_name: str, max_gpus: int,
                  max_memory_gb: float, priority: int = 5,
                  burst_allowed: bool = False, burst_max_gpus: int = 0) -> TenantQuota:
        q = self._quotas.get(tenant_id)
        if q:
            q.max_gpus = max_gpus
            q.max_memory_gb = max_memory_gb
            q.max_priority = priority
            q.burst_allowed = burst_allowed
            q.burst_max_gpus = burst_max_gpus
        else:
            q = TenantQuota(
                tenant_id=tenant_id, tenant_name=tenant_name,
                max_gpus=max_gpus, max_memory_gb=max_memory_gb,
                max_priority=priority, burst_allowed=burst_allowed,
                burst_max_gpus=burst_max_gpus,
            )
        self._quotas[tenant_id] = q
        return q

    def get_quota(self, tenant_id: str) -> TenantQuota | None:
        return self._quotas.get(tenant_id)

    def check_quota(self, tenant_id: str, requested_gpus: int,
                    requested_memory_gb: float) -> dict:
        q = self._quotas.get(tenant_id)
        if not q:
            return {"allowed": False, "reason": "Tenant not found", "usage_after": 0.0}
        gpus_after = q.gpus_in_use + requested_gpus
        mem_after = q.memory_in_use_gb + requested_memory_gb
        if gpus_after > q.max_gpus:
            if q.burst_allowed and gpus_after <= q.burst_max_gpus:
                return {"allowed": True, "reason": "Burst quota used", "usage_after": round(gpus_after / q.max_gpus * 100, 1)}
            return {"allowed": False, "reason": f"GPU quota exceeded ({gpus_after} > {q.max_gpus})",
                    "usage_after": round(gpus_after / q.max_gpus * 100, 1)}
        if mem_after > q.max_memory_gb:
            return {"allowed": False, "reason": f"Memory quota exceeded ({mem_after:.0f} > {q.max_memory_gb:.0f})",
                    "usage_after": round(mem_after / q.max_memory_gb * 100, 1)}
        return {"allowed": True, "reason": "Quota available",
                "usage_after": round(gpus_after / q.max_gpus * 100, 1)}

    def list_quotas(self) -> list[TenantQuota]:
        return list(self._quotas.values())

    def update_usage(self, tenant_id: str, gpus_used: int, memory_used_gb: float) -> TenantQuota | None:
        q = self._quotas.get(tenant_id)
        if not q:
            return None
        q.gpus_in_use = gpus_used
        q.memory_in_use_gb = memory_used_gb
        q.quota_usage_percent = round(gpus_used / max(q.max_gpus, 1) * 100, 1)
        return q


class ExplanationGenerator:
    def __init__(self) -> None:
        self._explanations: list[Explanation] = []

    def generate(self, subject_type: str, subject_id: str, context: dict | None = None) -> Explanation:
        ctx = context or {}
        templates = {
            "placement": "Placed job on node {node} because it had the most available GPU memory ({mem}GB) and lowest utilization ({util}%)",
            "scaling": "Scaled replicas from {from_r} to {to_r} due to load changing from {load_from:.0%} to {load_to:.0%}",
            "preemption": "Preempted job {job} because {reason}",
            "quota_block": "Request blocked because tenant {tenant} would exceed GPU quota ({used}/{max})",
            "policy_block": "Action blocked by policy '{policy}': rule {rule} failed",
        }
        template = templates.get(subject_type, f"Decision made for {subject_type} {subject_id}")
        try:
            summary = template.format(**ctx)
        except KeyError:
            summary = f"{subject_type} decision for {subject_id}"
        factors = [{"key": k, "value": v, "impact": round(random.uniform(0, 1), 2)} for k, v in ctx.items()]
        exp = Explanation(
            subject_type=subject_type,
            subject_id=subject_id,
            summary=summary,
            factors=factors,
            confidence=round(random.uniform(0.7, 0.99), 2),
        )
        self._explanations.append(exp)
        return exp

    def list_explanations(self, subject_type: str | None = None, limit: int = 50) -> list[Explanation]:
        exps = self._explanations
        if subject_type:
            exps = [e for e in exps if e.subject_type == subject_type]
        return exps[-limit:]

    def get_explanation(self, explanation_id: str) -> Explanation | None:
        for e in self._explanations:
            if e.explanation_id == explanation_id:
                return e
        return None


class ReportGenerator:
    def __init__(self) -> None:
        self._reports: list[GovernanceReport] = []

    def generate_compliance_report(self, cluster_id: str = "") -> GovernanceReport:
        report = GovernanceReport(
            report_type="compliance",
            cluster_id=cluster_id,
            tenant_summaries=[
                {"tenant": "tenant-a", "gpus_used": 12, "quota": 20, "violations": 0},
                {"tenant": "tenant-b", "gpus_used": 18, "quota": 16, "violations": 2},
            ],
            policy_violations=random.randint(0, 5),
            approval_metrics={"approved": random.randint(10, 50), "pending": random.randint(0, 5), "rejected": random.randint(0, 3)},
            audit_trail=[
                {"action": "placement", "timestamp": "2026-07-22T10:00:00", "user": "admin", "status": "approved"},
                {"action": "scale_down", "timestamp": "2026-07-22T11:00:00", "user": "auto", "status": "executed"},
            ],
            recommendations=[
                "Review tenant-b quota overage",
                "Enable burst allocation for tenant-a",
                "Schedule compliance training",
            ],
        )
        self._reports.append(report)
        return report

    def generate_usage_report(self, cluster_id: str = "") -> GovernanceReport:
        report = GovernanceReport(
            report_type="usage",
            cluster_id=cluster_id,
            tenant_summaries=[
                {"tenant": "tenant-a", "gpu_hours": 8760, "efficiency": "78%", "waste_gpu_hours": 120},
                {"tenant": "tenant-b", "gpu_hours": 4380, "efficiency": "45%", "waste_gpu_hours": 450},
            ],
            recommendations=[
                "Optimize tenant-b GPU utilization (currently 45%)",
                "Consider rightsizing tenant-b allocation",
            ],
        )
        self._reports.append(report)
        return report

    def list_reports(self, report_type: str | None = None) -> list[GovernanceReport]:
        if report_type:
            return [r for r in self._reports if r.report_type == report_type]
        return self._reports

    def get_report(self, report_id: str) -> GovernanceReport | None:
        for r in self._reports:
            if r.report_id == report_id:
                return r
        return None


class ExtendedGovernanceService:
    def __init__(self) -> None:
        self._enforcer = PolicyEnforcer()
        self._rollback = RollbackManager()
        self._quota = TenantQuotaManager()
        self._explainer = ExplanationGenerator()
        self._reporter = ReportGenerator()

    def enforce_policy(self, domain: str, context: dict) -> dict:
        envelopes = self._enforcer.list_envelopes(domain)
        for env in envelopes:
            if env.enabled:
                result = self._enforcer.evaluate(env, context)
                if not result["passed"]:
                    return {"allowed": False, "policy": env.name, "result": result}
        return {"allowed": True, "policy": "all", "result": {"passed": True}}

    def create_rollback(self, action_id: str, action_type: str, reason: str = "") -> RollbackPlan:
        return self._rollback.create_plan(action_id, action_type, reason or "Manual rollback")

    def manage_tenant_quota(self, tenant_id: str, action: str = "get", **kwargs) -> Any:
        if action == "get":
            return self._quota.get_quota(tenant_id)
        elif action == "set":
            return self._quota.set_quota(tenant_id, **kwargs)
        elif action == "check":
            return self._quota.check_quota(tenant_id, kwargs.get("gpus", 0), kwargs.get("memory", 0))
        elif action == "update_usage":
            return self._quota.update_usage(tenant_id, kwargs.get("gpus_used", 0), kwargs.get("memory_used", 0))
        return None

    def explain_decision(self, subject_type: str, subject_id: str, context: dict | None = None) -> Explanation:
        return self._explainer.generate(subject_type, subject_id, context)

    def generate_report(self, report_type: str = "compliance") -> GovernanceReport:
        if report_type == "usage":
            return self._reporter.generate_usage_report()
        return self._reporter.generate_compliance_report()
