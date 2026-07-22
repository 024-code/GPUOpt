from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _cleanup_global_state():
    from gpuopt.healing.router import _monitor_thread, _monitor_stop
    _monitor_stop.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=3)
    from gpuopt.registry import reset_registry
    reset_registry()
    yield


@pytest.fixture()
def client(tmp_path: Path):
    os.environ["GPUOPT_DATABASE_PATH"] = str(tmp_path / "test.db")
    from gpuopt.config import get_settings
    from gpuopt.dependencies import (
        get_actuation_service,
        get_alert_manager,
        get_analysis_service,
        get_anomaly_detector,
        get_approval_workflow,
        get_chaos_engine,
        get_check_service,
        get_compliance_engine,
        get_cost_analysis_service,
        get_dashboard_service,
        get_digital_twin,
        get_policy_engine,
        get_rec_engine,
        get_report_scheduler,
        get_repository,
        get_scheduler_service,
        get_state_service,
        get_tenant_manager,
        get_trace_service,
    )

    get_settings.cache_clear()
    get_repository.cache_clear()
    get_alert_manager.cache_clear()
    get_analysis_service.cache_clear()
    get_anomaly_detector.cache_clear()
    get_approval_workflow.cache_clear()
    get_chaos_engine.cache_clear()
    get_check_service.cache_clear()
    get_compliance_engine.cache_clear()
    get_cost_analysis_service.cache_clear()
    get_dashboard_service.cache_clear()
    get_digital_twin.cache_clear()
    get_policy_engine.cache_clear()
    get_rec_engine.cache_clear()
    get_report_scheduler.cache_clear()
    get_scheduler_service.cache_clear()
    get_state_service.cache_clear()
    get_tenant_manager.cache_clear()
    get_trace_service.cache_clear()

    from gpuopt.registry import reset_registry
    reset_registry()

    from gpuopt.main import app

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()
    get_repository.cache_clear()
    get_alert_manager.cache_clear()
    get_analysis_service.cache_clear()
    get_anomaly_detector.cache_clear()
    get_approval_workflow.cache_clear()
    get_chaos_engine.cache_clear()
    get_check_service.cache_clear()
    get_compliance_engine.cache_clear()
    get_cost_analysis_service.cache_clear()
    get_dashboard_service.cache_clear()
    get_digital_twin.cache_clear()
    get_policy_engine.cache_clear()
    get_rec_engine.cache_clear()
    get_report_scheduler.cache_clear()
    get_scheduler_service.cache_clear()
    get_state_service.cache_clear()
    get_tenant_manager.cache_clear()
    get_trace_service.cache_clear()
