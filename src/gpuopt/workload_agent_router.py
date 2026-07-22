from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from .schemas import WorkloadInput
from .workload_agent import WorkloadAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workload-agent", tags=["workload-agent"])
agent = WorkloadAgent()


@router.get("/detect")
async def detect_system(cluster_id: str = "") -> dict[str, Any]:
    system = agent.detect_system(cluster_id)
    return {"status": "ok", "system": system.model_dump(mode="json")}


@router.post("/submit")
async def submit_workload(workload: WorkloadInput) -> dict[str, Any]:
    return agent.submit_workload(workload)


@router.get("/assignments")
async def list_assignments() -> list[dict[str, Any]]:
    return [a.model_dump(mode="json") for a in agent.list_assignments()]


@router.get("/assignments/{assignment_id}")
async def get_assignment(assignment_id: str) -> dict[str, Any]:
    assignment = agent.get_assignment(assignment_id)
    if assignment is None:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return assignment.model_dump(mode="json")


@router.post("/assignments/{assignment_id}/complete")
async def complete_assignment(
    assignment_id: str,
    success: bool = True,
    duration_minutes: float = 0.0,
) -> dict[str, Any]:
    assignment = agent.complete_assignment(assignment_id, success, duration_minutes)
    if assignment is None:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return {"status": "completed", "assignment": assignment.model_dump(mode="json")}


@router.get("/stats")
async def get_stats() -> dict[str, Any]:
    return {"status": "ok", "stats": agent.get_stats()}
