from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter

from ..registry import get_registry
from .auto_healer import AutoHealer

logger = logging.getLogger(__name__)

healing_router = APIRouter(prefix="/api/v1/healing", tags=["auto-healing"])

_monitor_thread: threading.Thread | None = None
_monitor_stop = threading.Event()


def _get_healer() -> AutoHealer:
    reg = get_registry()
    return reg.get_or_create("healer", AutoHealer)


@healing_router.post("/check-health")
def check_health(telemetry: dict) -> dict:
    return _get_healer().check_node_health(telemetry)


@healing_router.post("/suggest")
def suggest_remediation(telemetry: dict, node_id: str = "unknown") -> dict:
    result = _get_healer().suggest_remediation(telemetry, node_id)
    if result is None:
        return {"status": "healthy", "message": "No remediation needed"}
    return {
        "node_id": result.node_id,
        "action": result.action.value,
        "status": result.status,
        "message": result.message,
    }


@healing_router.post("/execute")
def execute_remediation(telemetry: dict, node_id: str = "unknown") -> dict:
    result = _get_healer().execute_remediation(telemetry, node_id)
    return {
        "node_id": result.node_id,
        "action": result.action.value,
        "status": result.status,
        "message": result.message,
        "duration_seconds": result.duration_seconds,
        "timestamp": result.timestamp,
    }


@healing_router.get("/history")
def get_history(limit: int = 50) -> list[dict]:
    return _get_healer().get_history(limit)


@healing_router.get("/active")
def get_active_remediations() -> dict:
    return {"active_remediations": _get_healer().get_active_remediations()}


@healing_router.post("/clear")
def clear_history() -> dict:
    _get_healer().clear_history()
    return {"status": "cleared"}


@healing_router.get("/status")
def monitor_status() -> dict:
    global _monitor_thread
    running = _monitor_thread is not None and _monitor_thread.is_alive()
    return {
        "monitor_running": running,
        "active_remediations": _get_healer().get_active_remediations(),
        "total_remediations": len(_get_healer().remediation_history),
    }


@healing_router.post("/start")
def start_monitor(data: dict[str, Any]) -> dict:
    global _monitor_thread, _monitor_stop
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return {"status": "already_running", "message": "Monitor is already active"}

    cluster_id = data.get("cluster_id", "sandbox")
    interval = data.get("interval_seconds", 60)
    telemetry_fn = data.get("telemetry_fn")

    _monitor_stop.clear()

    def _loop() -> None:
        healer = _get_healer()
        while not _monitor_stop.is_set():
            try:
                mock_telemetry = {
                    "temperature": healer._rng.uniform(50, 95) if hasattr(healer, "_rng") else 70,
                    "ecc_errors": healer._rng.randint(0, 30) if hasattr(healer, "_rng") else 5,
                    "memory_utilization": healer._rng.uniform(60, 98) if hasattr(healer, "_rng") else 75,
                    "gpu_utilization": healer._rng.uniform(30, 100) if hasattr(healer, "_rng") else 60,
                    "xid_errors": healer._rng.randint(0, 15) if hasattr(healer, "_rng") else 2,
                }
                node_id = f"{cluster_id}-node-{healer._rng.randint(1, 10) if hasattr(healer, '_rng') else 1}"
                result = healer.execute_remediation(mock_telemetry, node_id)
                if result.status == "executed":
                    logger.info("Auto-healed %s on %s", result.action.value, node_id)
            except Exception:
                logger.exception("Auto-heal monitor cycle error")
            _monitor_stop.wait(interval)

    _monitor_thread = threading.Thread(target=_loop, daemon=True, name="heal-monitor")
    _monitor_thread.start()
    logger.info("Auto-healing monitor started for %s (interval=%ds)", cluster_id, interval)
    return {
        "status": "healing_started",
        "cluster_id": cluster_id,
        "interval_seconds": interval,
        "message": f"Auto-healing monitor active for {cluster_id}",
    }


@healing_router.post("/stop")
def stop_monitor() -> dict:
    global _monitor_thread, _monitor_stop
    if _monitor_thread is None or not _monitor_thread.is_alive():
        return {"status": "not_running", "message": "Monitor is not active"}
    _monitor_stop.set()
    _monitor_thread.join(timeout=10)
    _monitor_thread = None
    logger.info("Auto-healing monitor stopped")
    return {"status": "stopped", "message": "Auto-healing monitor stopped"}
