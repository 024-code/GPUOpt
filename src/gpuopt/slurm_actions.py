from __future__ import annotations

import copy
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .connectors.base import ClusterConnector
from .connectors.factory import build_connector
from .repository import ClusterRepository
from .schemas import (
    SlurmJobControlRequest,
    SlurmJobControlResult,
    SlurmReservation,
    SlurmReservationRequest,
)

logger = logging.getLogger(__name__)


class SlurmActionAdapter:
    """Performs real Slurm actions via CLI or pyslurm.

    Falls back to mock/simulated actions when Slurm CLI is unavailable
    and a mock cluster is configured.
    """

    _mock_state: dict[str, dict[str, Any]] = {}

    @classmethod
    def reset_mock_state(cls) -> None:
        cls._mock_state.clear()

    def __init__(self, repository: ClusterRepository) -> None:
        self._repository = repository

    def _mock_state_for(self, cluster_id: str) -> dict[str, Any]:
        if cluster_id not in self._mock_state:
            data = self._mock_data(cluster_id)
            self._mock_state[cluster_id] = copy.deepcopy(data)
        return self._mock_state[cluster_id]

    def _get_connector(self, cluster_id: str) -> ClusterConnector | None:
        from uuid import UUID
        cluster = self._repository.get_cluster(UUID(cluster_id))
        if cluster is None:
            return None
        return build_connector(cluster)

    def _run_cmd(self, cmd: list[str], timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                logger.warning("Slurm command %s failed (rc=%d): %s",
                               cmd[0], result.returncode, result.stderr.strip()[:200])
                return ""
            return result.stdout
        except FileNotFoundError:
            logger.warning("Slurm command not found: %s", cmd[0])
            return ""
        except subprocess.TimeoutExpired:
            logger.warning("Slurm command timed out: %s", cmd[0])
            return ""
        except Exception as exc:
            logger.warning("Slurm command error: %s", exc)
            return ""

    def _has_slurm_cli(self) -> bool:
        return bool(self._run_cmd(["sinfo", "--version"]))

    def _mock_data(self, cluster_id: str) -> dict[str, Any]:
        from uuid import UUID
        cluster = self._repository.get_cluster(UUID(cluster_id))
        if cluster is None:
            return {}
        path = cluster.options.get("mock_slurm_data", "")
        if not path:
            return {}
        mock_file = Path(path)
        if not mock_file.is_absolute():
            mock_file = Path(__file__).resolve().parents[2] / path
        if mock_file.exists():
            return json.loads(mock_file.read_text())
        return {}

    # ── Job Control ──────────────────────────────────────────

    def submit_job(self, cluster_id: str, req: SlurmJobControlRequest) -> SlurmJobControlResult:
        if self._has_slurm_cli():
            return self._submit_job_real(cluster_id, req)
        return self._submit_job_mock(cluster_id, req)

    def cancel_job(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        if self._has_slurm_cli():
            return self._cancel_job_real(cluster_id, job_id)
        return self._cancel_job_mock(cluster_id, job_id)

    def hold_job(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        if self._has_slurm_cli():
            return self._hold_job_real(cluster_id, job_id)
        return self._hold_job_mock(cluster_id, job_id)

    def release_job(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        if self._has_slurm_cli():
            return self._release_job_real(cluster_id, job_id)
        return self._release_job_mock(cluster_id, job_id)

    def modify_job(self, cluster_id: str, req: SlurmJobControlRequest) -> SlurmJobControlResult:
        if self._has_slurm_cli():
            return self._modify_job_real(cluster_id, req)
        return self._modify_job_mock(cluster_id, req)

    # ── Real job operations ──────────────────────────────────

    def _submit_job_real(self, cluster_id: str, req: SlurmJobControlRequest) -> SlurmJobControlResult:
        if req.script:
            output = self._run_cmd(["sbatch"], input=req.script)
        else:
            args = ["sbatch", "--parsable"]
            if req.partition:
                args.extend(["-p", req.partition])
            if req.gpu_count:
                args.extend(["--gpus", str(req.gpu_count)])
            if req.cpu_count:
                args.extend(["-c", str(req.cpu_count)])
            if req.memory_mb:
                args.extend(["--mem", f"{req.memory_mb}M"])
            if req.time_limit_minutes:
                args.extend(["-t", str(req.time_limit_minutes)])
            if req.node_count:
                args.extend(["-N", str(req.node_count)])
            if req.job_name:
                args.extend(["-J", req.job_name])
            if req.dependency:
                args.extend(["-d", req.dependency])
            args.append("--wrap", req.script or "/bin/hostname")
            output = self._run_cmd(args)
        if output:
            try:
                job_id = int(output.strip().split()[-1])
                return SlurmJobControlResult(success=True, action="submit", job_id=job_id, message=output.strip())
            except (ValueError, IndexError):
                pass
        return SlurmJobControlResult(success=False, action="submit", message="sbatch failed")

    def _cancel_job_real(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        output = self._run_cmd(["scancel", str(job_id)])
        return SlurmJobControlResult(
            success=True,
            action="cancel",
            job_id=job_id,
            message=f"Cancelled job {job_id}" if not output else output.strip(),
        )

    def _hold_job_real(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        output = self._run_cmd(["scontrol", "hold", str(job_id)])
        return SlurmJobControlResult(
            success=not output,
            action="hold",
            job_id=job_id,
            message=output.strip() if output else f"Job {job_id} held",
        )

    def _release_job_real(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        output = self._run_cmd(["scontrol", "release", str(job_id)])
        return SlurmJobControlResult(
            success=not output,
            action="release",
            job_id=job_id,
            message=output.strip() if output else f"Job {job_id} released",
        )

    def _modify_job_real(self, cluster_id: str, req: SlurmJobControlRequest) -> SlurmJobControlResult:
        args = ["scontrol", "update", f"jobid={req.job_id}"]
        if req.time_limit_minutes:
            args.append(f"timelimit={req.time_limit_minutes}")
        if req.partition:
            args.append(f"partition={req.partition}")
        output = self._run_cmd(args)
        return SlurmJobControlResult(
            success=not output,
            action="modify",
            job_id=req.job_id,
            message=output.strip() if output else f"Job {req.job_id} modified",
        )

    # ── Mock job operations ──────────────────────────────────

    def _submit_job_mock(self, cluster_id: str, req: SlurmJobControlRequest) -> SlurmJobControlResult:
        state = self._mock_state_for(cluster_id)
        all_jobs = state.get("running_jobs", []) + state.get("pending_jobs", [])
        max_id = max((j.get("job_id", 0) for j in all_jobs), default=1000)
        new_id = max_id + 1
        new_job = dict(job_id=new_id, job_name=req.job_name or f"mock_job_{new_id}",
                       partition=req.partition or "gpu", state="PENDING")
        state.setdefault("pending_jobs", []).append(new_job)
        return SlurmJobControlResult(
            success=True, action="submit", job_id=new_id,
            message=f"Mock job {new_id} submitted ({req.job_name or 'unnamed'})",
        )

    def _cancel_job_mock(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        state = self._mock_state_for(cluster_id)
        for pool in ("pending_jobs", "running_jobs"):
            for j in state.get(pool, []):
                if j.get("job_id") == job_id:
                    state[pool].remove(j)
                    j["state"] = "CANCELLED"
                    state.setdefault("cancelled_jobs", []).append(j)
                    return SlurmJobControlResult(
                        success=True, action="cancel", job_id=job_id,
                        message=f"Mock job {job_id} cancelled",
                    )
        return SlurmJobControlResult(
            success=False, action="cancel", job_id=job_id,
            message=f"Job {job_id} not found",
        )

    def _hold_job_mock(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        return SlurmJobControlResult(
            success=True, action="hold", job_id=job_id,
            message=f"Mock job {job_id} held",
        )

    def _release_job_mock(self, cluster_id: str, job_id: int) -> SlurmJobControlResult:
        return SlurmJobControlResult(
            success=True, action="release", job_id=job_id,
            message=f"Mock job {job_id} released",
        )

    def _modify_job_mock(self, cluster_id: str, req: SlurmJobControlRequest) -> SlurmJobControlResult:
        return SlurmJobControlResult(
            success=True, action="modify", job_id=req.job_id,
            message=f"Mock job {req.job_id} modified",
        )

    # ── Reservation Management ───────────────────────────────

    def create_reservation(self, cluster_id: str, req: SlurmReservationRequest) -> SlurmReservation:
        if self._has_slurm_cli():
            return self._create_reservation_real(cluster_id, req)
        return self._create_reservation_mock(cluster_id, req)

    def list_reservations(self, cluster_id: str) -> list[SlurmReservation]:
        if self._has_slurm_cli():
            return self._list_reservations_real(cluster_id)
        return self._list_reservations_mock(cluster_id)

    def delete_reservation(self, cluster_id: str, reservation_name: str) -> bool:
        if self._has_slurm_cli():
            return self._delete_reservation_real(cluster_id, reservation_name)
        return self._delete_reservation_mock(cluster_id, reservation_name)

    def _create_reservation_real(self, cluster_id: str, req: SlurmReservationRequest) -> SlurmReservation:
        args = ["scontrol", "create", f"reservationname={req.name}",
                f"starttime=now", f"duration={req.duration_minutes}"]
        if req.partition:
            args.append(f"partition={req.partition}")
        if req.node_count:
            args.append(f"nodes={req.node_count}")
        if req.users:
            args.append(f"users={','.join(req.users)}")
        if req.accounts:
            args.append(f"accounts={','.join(req.accounts)}")
        output = self._run_cmd(args)
        if output and "Reservation created" in output:
            return SlurmReservation(
                name=req.name,
                partition=req.partition,
                duration_minutes=req.duration_minutes,
                users=req.users,
                accounts=req.accounts,
                state="active",
            )
        return SlurmReservation(
            name=req.name, state="failed",
        )

    def _list_reservations_real(self, cluster_id: str) -> list[SlurmReservation]:
        output = self._run_cmd(["scontrol", "show", "reservation"])
        if not output:
            return []
        reservations: list[SlurmReservation] = []
        current: dict[str, Any] = {}
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                if current:
                    reservations.append(self._parse_reservation_line(current))
                    current = {}
                continue
            for part in line.split():
                if "=" in part:
                    key, val = part.split("=", 1)
                    current[key.lower()] = val
        if current:
            reservations.append(self._parse_reservation_line(current))
        return reservations

    def _parse_reservation_line(self, data: dict[str, Any]) -> SlurmReservation:
        return SlurmReservation(
            name=data.get("reservationname", ""),
            partition=data.get("partition", ""),
            nodes=data.get("nodes", "").split(",") if data.get("nodes") else [],
            duration_minutes=int(data.get("duration", 0)),
            users=data.get("users", "").split(",") if data.get("users") else [],
            accounts=data.get("accounts", "").split(",") if data.get("accounts") else [],
            state=data.get("state", "active"),
        )

    def _delete_reservation_real(self, cluster_id: str, reservation_name: str) -> bool:
        output = self._run_cmd(["scontrol", "delete", f"reservationname={reservation_name}"])
        return not output

    def _create_reservation_mock(self, cluster_id: str, req: SlurmReservationRequest) -> SlurmReservation:
        state = self._mock_state_for(cluster_id)
        reservation = SlurmReservation(
            name=req.name,
            partition=req.partition or "gpu",
            duration_minutes=req.duration_minutes,
            users=req.users,
            accounts=req.accounts,
            state="active",
        )
        state.setdefault("reservations", []).append(reservation.model_dump(mode="json"))
        return reservation

    def _list_reservations_mock(self, cluster_id: str) -> list[SlurmReservation]:
        state = self._mock_state_for(cluster_id)
        return [SlurmReservation(**r) for r in state.get("reservations", [])]

    def _delete_reservation_mock(self, cluster_id: str, reservation_name: str) -> bool:
        state = self._mock_state_for(cluster_id)
        reservations = state.get("reservations", [])
        for r in reservations:
            if r.get("name") == reservation_name:
                reservations.remove(r)
                return True
        return False

    # ── Accounting ────────────────────────────────────────────

    def get_job_accounting(self, cluster_id: str, job_id: int) -> dict[str, Any]:
        if self._has_slurm_cli():
            return self._get_job_accounting_real(cluster_id, job_id)
        return self._get_job_accounting_mock(cluster_id, job_id)

    def _get_job_accounting_real(self, cluster_id: str, job_id: int) -> dict[str, Any]:
        output = self._run_cmd([
            "sacct", "-j", str(job_id),
            "--format", "JobID,JobName,Partition,Account,User,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ExitCode",
            "--noheader", "-P",
        ], timeout=15)
        if not output:
            return {"job_id": job_id, "error": "No accounting data"}
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        entries = []
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            entries.append({
                "job_id": parts[0] if len(parts) > 0 else "",
                "name": parts[1] if len(parts) > 1 else "",
                "partition": parts[2] if len(parts) > 2 else "",
                "account": parts[3] if len(parts) > 3 else "",
                "user": parts[4] if len(parts) > 4 else "",
                "state": parts[5] if len(parts) > 5 else "",
                "elapsed": parts[6] if len(parts) > 6 else "",
                "total_cpu": parts[7] if len(parts) > 7 else "",
                "max_rss": parts[8] if len(parts) > 8 else "",
                "max_vmsize": parts[9] if len(parts) > 9 else "",
                "exit_code": parts[10] if len(parts) > 10 else "",
            })
        return {"job_id": job_id, "entries": entries}

    def _get_job_accounting_mock(self, cluster_id: str, job_id: int) -> dict[str, Any]:
        data = self._mock_data(cluster_id)
        for pool in ("running_jobs", "pending_jobs"):
            for j in data.get(pool, []):
                if j.get("job_id") == job_id:
                    return {
                        "job_id": job_id,
                        "entries": [{
                            "job_id": str(job_id),
                            "name": j.get("job_name", ""),
                            "partition": j.get("partition", ""),
                            "state": j.get("state", ""),
                            "account": "mock",
                            "user": j.get("user", ""),
                            "elapsed": f"{j.get('time_used_minutes', 0)}:00",
                            "total_cpu": str(j.get("cpus", 0)),
                            "max_rss": "",
                            "exit_code": "0:0",
                        }],
                    }
        return {"job_id": job_id, "entries": []}
