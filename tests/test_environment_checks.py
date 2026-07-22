from __future__ import annotations

from gpuopt.environment_checks import EnvironmentChecksService
from gpuopt.environment_checks_schemas import (
    CheckResult,
    CheckSeverity,
    CheckStatus,
    EnvironmentCheckCatalog,
    EnvironmentCheckRun,
    EnvironmentType,
    MandatoryCheck,
)


# ── Unit: Catalog ────────────────────────────────────────────

def test_catalog_all_environments():
    svc = EnvironmentChecksService()
    catalog = svc.get_catalog()
    assert len(catalog) == 3
    envs = [c.environment for c in catalog]
    assert EnvironmentType.SANDBOX in envs
    assert EnvironmentType.STAGING in envs
    assert EnvironmentType.PRODUCTION in envs


def test_catalog_sandbox():
    svc = EnvironmentChecksService()
    catalog = svc.get_catalog(EnvironmentType.SANDBOX)
    assert len(catalog) == 1
    assert catalog[0].environment == EnvironmentType.SANDBOX
    assert len(catalog[0].checks) == 10


def test_catalog_staging():
    svc = EnvironmentChecksService()
    catalog = svc.get_catalog(EnvironmentType.STAGING)
    assert len(catalog[0].checks) == 10


def test_catalog_production():
    svc = EnvironmentChecksService()
    catalog = svc.get_catalog(EnvironmentType.PRODUCTION)
    assert len(catalog[0].checks) == 10


def test_sandbox_check_names():
    svc = EnvironmentChecksService()
    catalog = svc.get_catalog(EnvironmentType.SANDBOX)
    names = [c.check_name for c in catalog[0].checks]
    expected = ["API server", "RBAC", "Nodes", "GPU inventory", "NVIDIA stack",
                 "DCGM exporter", "Prometheus", "Queueing", "Network and DNS", "Clock and certificates"]
    assert names == expected


def test_sandbox_severities():
    svc = EnvironmentChecksService()
    catalog = svc.get_catalog(EnvironmentType.SANDBOX)
    severities = {c.check_name: c.severity for c in catalog[0].checks}
    assert severities["API server"] == CheckSeverity.FAIL
    assert severities["Nodes"] == CheckSeverity.FAIL_OR_WARNING
    assert severities["Prometheus"] == CheckSeverity.WARNING_THEN_FAIL_BEFORE_R02
    assert severities["Clock and certificates"] == CheckSeverity.WARNING_THEN_FAIL_BEFORE_STAGING
    assert severities["Queueing"] == CheckSeverity.WARNING


# ── Unit: Run Checks ─────────────────────────────────────────

def test_run_sandbox():
    svc = EnvironmentChecksService()
    result = svc.run_checks(EnvironmentType.SANDBOX)
    assert isinstance(result, EnvironmentCheckRun)
    assert result.environment == EnvironmentType.SANDBOX
    assert len(result.checks) == 10
    assert result.passed_count + result.failed_count + result.warning_count == 10


def test_run_staging():
    svc = EnvironmentChecksService()
    result = svc.run_checks(EnvironmentType.STAGING)
    assert result.environment == EnvironmentType.STAGING
    assert result.passed_count > 0


def test_run_production():
    svc = EnvironmentChecksService()
    result = svc.run_checks(EnvironmentType.PRODUCTION)
    assert result.environment == EnvironmentType.PRODUCTION
    assert result.passed_count == 10
    assert result.overall_pass is True


def test_sandbox_specific_results():
    svc = EnvironmentChecksService()
    result = svc.run_checks(EnvironmentType.SANDBOX)
    results = {r.check_name: r for r in result.checks}
    assert results["API server"].status == CheckStatus.PASS
    assert results["RBAC"].status == CheckStatus.PASS
    assert results["Nodes"].status == CheckStatus.WARNING
    assert results["GPU inventory"].status == CheckStatus.FAIL
    assert results["NVIDIA stack"].status == CheckStatus.FAIL
    assert results["DCGM exporter"].status == CheckStatus.FAIL
    assert results["Prometheus"].status == CheckStatus.WARNING
    assert results["Queueing"].status == CheckStatus.WARNING
    assert results["Network and DNS"].status == CheckStatus.FAIL
    assert results["Clock and certificates"].status == CheckStatus.WARNING


def test_sandbox_summary():
    svc = EnvironmentChecksService()
    result = svc.run_checks(EnvironmentType.SANDBOX)
    assert "sandbox" in result.summary
    assert "Passed:" in result.summary
    assert "Failed:" in result.summary


def test_sandbox_overall_pass():
    svc = EnvironmentChecksService()
    result = svc.run_checks(EnvironmentType.SANDBOX)
    assert result.overall_pass is False  # sandbox has expected FAIL checks (GPU, NVIDIA, DCGM, Network)


def test_health():
    svc = EnvironmentChecksService()
    h = svc.health()
    assert h["status"] == "healthy"
    assert h["environments_defined"] == 3


# ── API Tests ─────────────────────────────────────────────────

def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/environment-checks/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


def test_catalog_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/environment-checks/catalog")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3


def test_catalog_api_filtered():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/environment-checks/catalog?environment=sandbox")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["environment"] == "sandbox"
        assert len(data[0]["checks"]) == 10


def test_run_api_default():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/environment-checks/run")
        assert r.status_code == 200
        data = r.json()
        assert data["environment"] == "sandbox"
        assert len(data["checks"]) == 10


def test_run_api_staging():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/environment-checks/run", json={"environment": "staging"})
        assert r.status_code == 200
        data = r.json()
        assert data["environment"] == "staging"
        assert data["passed_count"] > 0


def test_run_api_production():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/environment-checks/run", json={"environment": "production"})
        assert r.status_code == 200
        data = r.json()
        assert data["environment"] == "production"
        assert data["overall_pass"] is True
        assert data["passed_count"] == 10
