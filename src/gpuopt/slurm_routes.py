from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_repository
from .rbac import Permission, require_permission
from .repository import ClusterRepository
from .schemas import (
    SlurmJobControlRequest,
    SlurmJobControlResult,
    SlurmReservation,
    SlurmReservationRequest,
)
from .slurm_actions import SlurmActionAdapter

slurm_router = APIRouter(prefix="/api/v1/slurm", tags=["slurm"])


def _get_adapter(repository: ClusterRepository = Depends(get_repository)) -> SlurmActionAdapter:
    return SlurmActionAdapter(repository)


@slurm_router.post("/{cluster_id}/jobs/submit")
def submit_job(
    cluster_id: UUID,
    req: SlurmJobControlRequest,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> SlurmJobControlResult:
    return adapter.submit_job(str(cluster_id), req)


@slurm_router.post("/{cluster_id}/jobs/{job_id}/cancel")
def cancel_job(
    cluster_id: UUID,
    job_id: int,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> SlurmJobControlResult:
    return adapter.cancel_job(str(cluster_id), job_id)


@slurm_router.post("/{cluster_id}/jobs/{job_id}/hold")
def hold_job(
    cluster_id: UUID,
    job_id: int,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> SlurmJobControlResult:
    return adapter.hold_job(str(cluster_id), job_id)


@slurm_router.post("/{cluster_id}/jobs/{job_id}/release")
def release_job(
    cluster_id: UUID,
    job_id: int,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> SlurmJobControlResult:
    return adapter.release_job(str(cluster_id), job_id)


@slurm_router.patch("/{cluster_id}/jobs/{job_id}")
def modify_job(
    cluster_id: UUID,
    job_id: int,
    req: SlurmJobControlRequest,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> SlurmJobControlResult:
    req.job_id = job_id
    req.action = "modify"
    return adapter.modify_job(str(cluster_id), req)


@slurm_router.post("/{cluster_id}/reservations")
def create_reservation(
    cluster_id: UUID,
    req: SlurmReservationRequest,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> SlurmReservation:
    return adapter.create_reservation(str(cluster_id), req)


@slurm_router.get("/{cluster_id}/reservations")
def list_reservations(
    cluster_id: UUID,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.CLUSTER_READ)),
) -> list[SlurmReservation]:
    return adapter.list_reservations(str(cluster_id))


@slurm_router.delete("/{cluster_id}/reservations/{reservation_name}")
def delete_reservation(
    cluster_id: UUID,
    reservation_name: str,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> dict:
    if not adapter.delete_reservation(str(cluster_id), reservation_name):
        raise HTTPException(404, "Reservation not found")
    return {"status": "deleted", "reservation": reservation_name}


@slurm_router.get("/{cluster_id}/accounting/{job_id}")
def get_job_accounting(
    cluster_id: UUID,
    job_id: int,
    adapter: SlurmActionAdapter = Depends(_get_adapter),
    _: None = Depends(require_permission(Permission.CLUSTER_READ)),
) -> dict:
    return adapter.get_job_accounting(str(cluster_id), job_id)
