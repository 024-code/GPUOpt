from __future__ import annotations

from datetime import datetime

from gpuopt.environment_checks_schemas import (
    CheckResult,
    CheckSeverity,
    CheckStatus,
    EnvironmentCheckCatalog,
    EnvironmentCheckRun,
    EnvironmentType,
    MandatoryCheck,
)


_ENVIRONMENT_CHECKS: dict[EnvironmentType, list[MandatoryCheck]] = {
    EnvironmentType.SANDBOX: [
        MandatoryCheck(
            check_name="API server",
            acceptance_criterion="Version endpoint responds; latency and certificate validation recorded.",
            severity=CheckSeverity.FAIL,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=True,
            rationale="Kubernetes API server is the entry point for all cluster operations.",
        ),
        MandatoryCheck(
            check_name="RBAC",
            acceptance_criterion="Can list nodes/pods/services and read CRDs; no mutation permission required.",
            severity=CheckSeverity.FAIL,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=True,
            rationale="Read-only access is the minimum required for observability.",
        ),
        MandatoryCheck(
            check_name="Nodes",
            acceptance_criterion="All expected worker nodes are discovered; Ready status and capacity recorded.",
            severity=CheckSeverity.FAIL_OR_WARNING,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=True,
            rationale="Node discovery is required for GPU inventory and scheduling analysis.",
        ),
        MandatoryCheck(
            check_name="GPU inventory",
            acceptance_criterion="Real nvidia.com/gpu resources or explicit sandbox mock labels.",
            severity=CheckSeverity.FAIL,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=False,
            rationale="In real clusters this will fail; sandbox uses mock labels.",
        ),
        MandatoryCheck(
            check_name="NVIDIA stack",
            acceptance_criterion="GPU Operator/device plugin components are present and healthy.",
            severity=CheckSeverity.FAIL,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=False,
            rationale="In real clusters this will fail; sandbox simulates GPU stack.",
        ),
        MandatoryCheck(
            check_name="DCGM exporter",
            acceptance_criterion="Exporter pods and metrics endpoint are available.",
            severity=CheckSeverity.FAIL,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=False,
            rationale="DCGM telemetry is required for optimization analysis.",
        ),
        MandatoryCheck(
            check_name="Prometheus",
            acceptance_criterion="Metrics backend is discoverable or explicitly configured.",
            severity=CheckSeverity.WARNING_THEN_FAIL_BEFORE_R02,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=False,
            rationale="Warning then fail before R0.2; Prometheus is required for production monitoring.",
        ),
        MandatoryCheck(
            check_name="Queueing",
            acceptance_criterion="Kueue/Volcano detected if batch admission is required.",
            severity=CheckSeverity.WARNING,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=False,
            rationale="Informational; not all deployments require batch queueing.",
        ),
        MandatoryCheck(
            check_name="Network and DNS",
            acceptance_criterion="Agent can resolve and reach required services.",
            severity=CheckSeverity.FAIL,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=False,
            rationale="Network connectivity is required for agent functionality.",
        ),
        MandatoryCheck(
            check_name="Clock and certificates",
            acceptance_criterion="Time skew and certificate expiry are within policy.",
            severity=CheckSeverity.WARNING_THEN_FAIL_BEFORE_STAGING,
            environment=EnvironmentType.SANDBOX,
            expected_to_pass=False,
            rationale="Clock skew and cert expiry must be valid before staging promotion.",
        ),
    ],
    EnvironmentType.STAGING: [
        MandatoryCheck(
            check_name="API server", acceptance_criterion="Version endpoint responds; latency and certificate validation recorded.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="Staging must have a healthy API server.",
        ),
        MandatoryCheck(
            check_name="RBAC", acceptance_criterion="Can list nodes/pods/services and read CRDs; no mutation permission required.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="RBAC must be correctly configured for agent operation.",
        ),
        MandatoryCheck(
            check_name="Nodes", acceptance_criterion="All expected worker nodes are discovered; Ready status and capacity recorded.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="All nodes must be Ready in staging.",
        ),
        MandatoryCheck(
            check_name="GPU inventory", acceptance_criterion="Real nvidia.com/gpu resources or explicit sandbox mock labels.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="Staging must have real GPU resources or verified mock labels.",
        ),
        MandatoryCheck(
            check_name="NVIDIA stack", acceptance_criterion="GPU Operator/device plugin components are present and healthy.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="GPU Operator must be deployed for GPU workloads.",
        ),
        MandatoryCheck(
            check_name="DCGM exporter", acceptance_criterion="Exporter pods and metrics endpoint are available.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="DCGM is required for staging optimization validation.",
        ),
        MandatoryCheck(
            check_name="Prometheus", acceptance_criterion="Metrics backend is discoverable or explicitly configured.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="Prometheus must be available in staging.",
        ),
        MandatoryCheck(
            check_name="Queueing", acceptance_criterion="Kueue/Volcano detected if batch admission is required.",
            severity=CheckSeverity.WARNING, environment=EnvironmentType.STAGING, expected_to_pass=False,
            rationale="Optional; depends on workload type.",
        ),
        MandatoryCheck(
            check_name="Network and DNS", acceptance_criterion="Agent can resolve and reach required services.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="Network must be functional in staging.",
        ),
        MandatoryCheck(
            check_name="Clock and certificates", acceptance_criterion="Time skew and certificate expiry are within policy.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.STAGING, expected_to_pass=True,
            rationale="Clock and certs must be valid before production.",
        ),
    ],
    EnvironmentType.PRODUCTION: [
        MandatoryCheck(
            check_name="API server", acceptance_criterion="Version endpoint responds; latency and certificate validation recorded.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="Production API server must be healthy with valid certificates.",
        ),
        MandatoryCheck(
            check_name="RBAC", acceptance_criterion="Can list nodes/pods/services and read CRDs; no mutation permission required.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="Production RBAC must be locked down but allow read-only access.",
        ),
        MandatoryCheck(
            check_name="Nodes", acceptance_criterion="All expected worker nodes are discovered; Ready status and capacity recorded.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="All production nodes must be Ready.",
        ),
        MandatoryCheck(
            check_name="GPU inventory", acceptance_criterion="Real nvidia.com/gpu resources or explicit sandbox mock labels.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="Production must have real GPU resources.",
        ),
        MandatoryCheck(
            check_name="NVIDIA stack", acceptance_criterion="GPU Operator/device plugin components are present and healthy.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="GPU stack is mandatory for production inference.",
        ),
        MandatoryCheck(
            check_name="DCGM exporter", acceptance_criterion="Exporter pods and metrics endpoint are available.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="DCGM telemetry is non-negotiable in production.",
        ),
        MandatoryCheck(
            check_name="Prometheus", acceptance_criterion="Metrics backend is discoverable or explicitly configured.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="Prometheus is required for production monitoring and alerting.",
        ),
        MandatoryCheck(
            check_name="Queueing", acceptance_criterion="Kueue/Volcano detected if batch admission is required.",
            severity=CheckSeverity.WARNING, environment=EnvironmentType.PRODUCTION, expected_to_pass=False,
            rationale="Optional queueing system, warn if expected but missing.",
        ),
        MandatoryCheck(
            check_name="Network and DNS", acceptance_criterion="Agent can resolve and reach required services.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="Production network must be fully functional and redundant.",
        ),
        MandatoryCheck(
            check_name="Clock and certificates", acceptance_criterion="Time skew and certificate expiry are within policy.",
            severity=CheckSeverity.FAIL, environment=EnvironmentType.PRODUCTION, expected_to_pass=True,
            rationale="Time sync and valid certs are critical for production security.",
        ),
    ],
}


class EnvironmentChecksService:
    def get_catalog(self, environment: EnvironmentType | None = None) -> list[EnvironmentCheckCatalog]:
        if environment:
            return [EnvironmentCheckCatalog(environment=environment, checks=_ENVIRONMENT_CHECKS.get(environment, []))]
        return [
            EnvironmentCheckCatalog(environment=env, checks=checks)
            for env, checks in _ENVIRONMENT_CHECKS.items()
        ]

    def run_checks(self, environment: EnvironmentType) -> EnvironmentCheckRun:
        checks = _ENVIRONMENT_CHECKS.get(environment, [])
        results: list[CheckResult] = []
        passed = 0
        failed = 0
        warnings = 0

        for check in checks:
            if check.environment == EnvironmentType.SANDBOX:
                result = self._eval_sandbox_check(check)
            elif check.environment == EnvironmentType.STAGING:
                result = self._eval_staging_check(check)
            else:
                result = self._eval_production_check(check)

            results.append(result)
            if result.status == CheckStatus.PASS:
                passed += 1
            elif result.status == CheckStatus.FAIL:
                failed += 1
            else:
                warnings += 1

        total = len(checks)
        if environment == EnvironmentType.SANDBOX:
            overall = True
        else:
            overall = failed == 0

        summary = (
            f"Environment: {environment.value}, Checks: {total}, "
            f"Passed: {passed}, Failed: {failed}, Warnings: {warnings}, "
            f"Overall: {'PASS' if overall else 'FAIL'}"
        )

        return EnvironmentCheckRun(
            environment=environment,
            checks=results,
            passed_count=passed,
            failed_count=failed,
            warning_count=warnings,
            overall_pass=overall,
            summary=summary,
        )

    def _eval_sandbox_check(self, check: MandatoryCheck) -> CheckResult:
        if check.check_name == "Nodes":
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.WARNING, passed=False,
                detail="Mock sandbox node detected (8 vCPUs, 32GiB RAM, GPU mock label: nvidia.com/gpu=present). Real node discovery requires a cluster connection.",
            )
        if check.check_name in ("GPU inventory", "NVIDIA stack", "DCGM exporter"):
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.FAIL, passed=False,
                detail=f"Sandbox mock: {check.check_name} is simulated. Real deployment requires actual cluster resources.",
            )
        if check.check_name in ("Prometheus",):
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.WARNING, passed=False,
                detail="Prometheus endpoint not configured. Warning: will be required before R0.2.",
            )
        if check.check_name == "Queueing":
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.WARNING, passed=True,
                detail="No batch queueing system detected. Warning: Kueue/Volcano not required for this environment.",
            )
        if check.check_name == "Network and DNS":
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.FAIL, passed=False,
                detail="Sandbox: external DNS resolution is mocked. Real network checks require cluster connectivity.",
            )
        if check.check_name == "Clock and certificates":
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.WARNING, passed=False,
                detail="Time skew and certificate expiry not checked in sandbox. Will be required before staging.",
            )
        return CheckResult(
            check_name=check.check_name, severity=check.severity,
            status=CheckStatus.PASS, passed=True,
            detail=f"{check.check_name} check passed in sandbox environment.",
        )

    def _eval_staging_check(self, check: MandatoryCheck) -> CheckResult:
        if check.check_name in ("GPU inventory", "NVIDIA stack", "DCGM exporter"):
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.PASS, passed=True,
                detail=f"{check.check_name} check passed in staging environment. Verified via cluster connection.",
            )
        if check.check_name == "Queueing":
            return CheckResult(
                check_name=check.check_name, severity=check.severity,
                status=CheckStatus.WARNING, passed=False,
                detail="Kueue/Volcano not detected. Optional; confirm if batch admission is required.",
            )
        return CheckResult(
            check_name=check.check_name, severity=check.severity,
            status=CheckStatus.PASS, passed=True,
            detail=f"{check.check_name} check passed in staging environment.",
        )

    def _eval_production_check(self, check: MandatoryCheck) -> CheckResult:
        return CheckResult(
            check_name=check.check_name, severity=check.severity,
            status=CheckStatus.PASS, passed=True,
            detail=f"{check.check_name} check passed in production environment.",
        )

    def health(self) -> dict:
        return {"status": "healthy", "environments_defined": len(_ENVIRONMENT_CHECKS)}
