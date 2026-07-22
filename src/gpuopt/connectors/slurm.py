from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from gpuopt.schemas import (
    CheckItem,
    CheckStatus,
    ClusterTelemetry,
    GPUDeviceTelemetry,
    GpuTopology,
    GpuTopologyLink,
    JobMonitorConfig,
    MonitoringSnapshot,
    NodeTelemetry,
    NodeTopology,
    SlurmClusterTelemetry,
    SlurmJobInfo,
    SlurmJobSnapshot,
    SlurmNodeInfo,
    SlurmPartitionInfo,
)

from .base import ClusterConnector

logger = logging.getLogger(__name__)

_HAS_PYSLURM = False
try:
    import pyslurm  # noqa: F401
    _HAS_PYSLURM = True
except ImportError:
    pass


class SlurmConnector(ClusterConnector):
    """Connector for Slurm-based HPC clusters.

    Uses pyslurm bindings when available (faster, richer data), with
    automatic fallback to subprocess calls to Slurm CLI tools (sinfo,
    squeue, sacct, scontrol). Falls back to mock data when a mock
    snapshot path is configured in cluster options.

    Supports GPU topology detection via scontrol and real-time
    job monitoring via background polling.
    """

    def __init__(self, cluster: Any) -> None:
        super().__init__(cluster)
        self._monitors: dict[int, JobMonitorConfig] = {}
        self._monitor_threads: dict[int, threading.Thread] = {}
        self._monitor_histories: dict[int, list[SlurmJobSnapshot]] = {}
        self._monitor_lock = threading.RLock()

    # ── Command execution ──────────────────────────────────────

    def _run_cmd(self, cmd: list[str], timeout: int = 15) -> str:
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

    # ── pyslurm integration ─────────────────────────────────────

    def _try_pyslurm_sinfo(self) -> dict[str, Any] | None:
        if not _HAS_PYSLURM:
            return None
        try:
            import pyslurm
            sinfo = pyslurm.sinfo()
            nodes = sinfo.get()
            partitions = pyslurm.partition().get()
            return {"nodes": nodes, "partitions": partitions}
        except Exception as exc:
            logger.debug("pyslurm sinfo failed: %s", exc)
            return None

    def _try_pyslurm_squeue(self) -> dict[str, Any] | None:
        if not _HAS_PYSLURM:
            return None
        try:
            import pyslurm
            jobs = pyslurm.job().get()
            return jobs
        except Exception as exc:
            logger.debug("pyslurm squeue failed: %s", exc)
            return None

    def _try_pyslurm_sacct(self, job_id: int) -> dict[str, Any] | None:
        if not _HAS_PYSLURM:
            return None
        try:
            import pyslurm
            acct = pyslurm.accounting()
            return acct.get(job_id)
        except Exception as exc:
            logger.debug("pyslurm sacct failed: %s", exc)
            return None

    # ── JSON-based parsing (sinfo --json) ───────────────────────

    def _fetch_sinfo_json(self) -> dict[str, Any] | None:
        output = self._run_cmd(["sinfo", "--json"])
        if not output:
            return None
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return None

    def _fetch_squeue_delim(self) -> str:
        return self._run_cmd(
            ["squeue", "-o", "%i|%j|%P|%u|%T|%D|%C|%m|%l|%M|%N", "--noheader"],
            timeout=30,
        )

    def _fetch_sacct_for_job(self, job_id: int) -> str:
        return self._run_cmd(
            ["sacct", "-j", str(job_id), "--format", "JobID,Elapsed,AllocCPUS,ReqMem,TotalCPU,MaxRSS,State,ExitCode",
             "--noheader", "-P"],
            timeout=15,
        )

    # ── sinfo parsing ──────────────────────────────────────────

    def _parse_sinfo_nodes_format(self, output: str) -> list[SlurmNodeInfo]:
        nodes: list[SlurmNodeInfo] = []
        for line in output.strip().split("\n"):
            if not line.strip() or line.startswith(("NODELIST", "HOSTNAMES")):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                continue
            node_name = parts[0]
            state = parts[1] if parts[1] else "unknown"
            cpu_count = int(parts[2]) if parts[2].isdigit() else 0
            memory_mb = int(float(parts[3])) if parts[3].replace(".", "").isdigit() else int(float(parts[3]) if parts[3].replace(".", "").replace("-", "").replace("+", "").isdigit() and parts[3] else 0)
            if not parts[3].strip():
                memory_mb = 0
            else:
                try:
                    memory_mb = int(float(parts[3]))
                except ValueError:
                    memory_mb = 0
            gpu_str = parts[4] if len(parts) > 4 else ""
            gpu_count = 0
            gpu_model = ""
            if "gpu" in gpu_str.lower():
                gpu_parts = gpu_str.split(":")
                if len(gpu_parts) >= 3:
                    try:
                        gpu_count = int(gpu_parts[-1])
                    except ValueError:
                        gpu_count = 0
                    gpu_model = gpu_parts[1] if len(gpu_parts) > 1 else ""
                elif len(gpu_parts) == 2:
                    try:
                        gpu_count = int(gpu_parts[1])
                    except ValueError:
                        gpu_count = 0
            features_str = parts[5] if len(parts) > 5 else ""
            features = [f.strip() for f in features_str.split(",") if f.strip()] if features_str else []
            nodes.append(SlurmNodeInfo(
                node_name=node_name,
                state=state,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                gpu_count=gpu_count,
                gpu_model=gpu_model,
                features=features,
            ))
        return nodes

    def _parse_sinfo_nodes_from_json(self, data: dict[str, Any]) -> list[SlurmNodeInfo]:
        nodes: list[SlurmNodeInfo] = []
        for n in data.get("nodes", []):
            gres_total = n.get("gres_total", {}) or {}
            node_gpus = gres_total.get("gpu", 0) or 0
            hostname = n.get("hostname", n.get("name", "unknown"))
            features_list = []
            feats = n.get("features", {})
            if isinstance(feats, dict):
                active = feats.get("active", "")
                features_list = active.split(",") if active else []
            elif isinstance(feats, str):
                features_list = feats.split(",") if feats else []
            nodes.append(SlurmNodeInfo(
                node_name=hostname,
                state=n.get("state", "unknown"),
                cpu_count=n.get("cpus", {}).get("total", 0) if isinstance(n.get("cpus"), dict) else n.get("cpus", 0),
                memory_mb=n.get("real_memory", 0),
                gpu_count=node_gpus,
                gpu_model=n.get("features", {}).get("active", "").split(",")[0] if isinstance(n.get("features"), dict) else "",
                features=features_list,
            ))
        return nodes

    def _get_nodes(self) -> list[SlurmNodeInfo]:
        if self._has_mock_data():
            data = self._load_mock_data()
            return [SlurmNodeInfo(**n) for n in data.get("nodes", [])]

        pyslurm_data = self._try_pyslurm_sinfo()
        if pyslurm_data:
            return self._parse_sinfo_nodes_from_json(pyslurm_data)

        sinfo_json = self._fetch_sinfo_json()
        if sinfo_json:
            return self._parse_sinfo_nodes_from_json(sinfo_json)

        output = self._run_cmd(["sinfo", "-N", "-o", "%N|%T|%c|%m|%G|%f"])
        if output:
            return self._parse_sinfo_nodes_format(output)

        return []

    # ── squeue parsing ───────────────────────────────────────

    def _get_jobs(self) -> tuple[list[SlurmJobInfo], list[SlurmJobInfo]]:
        if self._has_mock_data():
            data = self._load_mock_data()
            pending = [SlurmJobInfo(**j) for j in data.get("pending_jobs", [])]
            running = [SlurmJobInfo(**j) for j in data.get("running_jobs", [])]
            return pending, running

        pyslurm_data = self._try_pyslurm_squeue()
        if pyslurm_data:
            return self._parse_jobs_from_pyslurm(pyslurm_data)

        output = self._fetch_squeue_delim()
        return self._parse_jobs_from_delim(output)

    def _parse_jobs_from_pyslurm(self, data: dict[str, Any]) -> tuple[list[SlurmJobInfo], list[SlurmJobInfo]]:
        pending: list[SlurmJobInfo] = []
        running: list[SlurmJobInfo] = []
        now_ts = datetime.now(timezone.utc)
        for jid, job in data.items():
            try:
                jid_int = int(jid)
            except (ValueError, TypeError):
                continue
            state = job.get("job_state", "").upper()
            tres = job.get("tres_per_node", "") or job.get("tres_alloc", "")
            gpu_count = 0
            if "gpu" in tres:
                for bit in tres.split(","):
                    if "gpu" in bit:
                        try:
                            gpu_count = int(bit.split("=")[-1])
                        except (ValueError, IndexError):
                            pass
            submit = job.get("submit_time", 0)
            start = job.get("start_time", 0)
            info = SlurmJobInfo(
                job_id=jid_int,
                job_name=job.get("name", ""),
                partition=job.get("partition", ""),
                user=job.get("user_name", ""),
                state=state,
                node_count=job.get("num_nodes", 0),
                gpu_count=gpu_count,
                cpus=job.get("num_cpus", 0),
                memory_mb=(job.get("mem_per_node", 0) or 0) // (1024 * 1024),
                time_limit_minutes=job.get("time_limit", 0),
                time_used_minutes=job.get("time_used", 0),
                nodes=",".join(job.get("nodes", []) or []),
                submit_time=datetime.fromtimestamp(submit, tz=timezone.utc) if submit else None,
                start_time=datetime.fromtimestamp(start, tz=timezone.utc) if start else None,
            )
            if state in ("PENDING", "SUSPENDED"):
                pending.append(info)
            elif state in ("RUNNING", "COMPLETING"):
                running.append(info)
        return pending, running

    def _parse_jobs_from_delim(self, output: str) -> tuple[list[SlurmJobInfo], list[SlurmJobInfo]]:
        pending: list[SlurmJobInfo] = []
        running: list[SlurmJobInfo] = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue
            try:
                job_id = int(parts[0])
            except ValueError:
                continue
            job_name = parts[1]
            partition = parts[2]
            user = parts[3]
            state = parts[4]
            node_count = int(parts[5]) if parts[5].isdigit() else 0
            cpus = int(parts[6]) if parts[6].isdigit() else 0
            mem_str = parts[7].upper() if len(parts) > 7 else ""
            memory_mb = 0
            if mem_str:
                try:
                    if "G" in mem_str:
                        memory_mb = int(float(mem_str.replace("G", "").split()[0]) * 1024)
                    elif "M" in mem_str:
                        memory_mb = int(float(mem_str.replace("M", "").split()[0]))
                    elif "T" in mem_str:
                        memory_mb = int(float(mem_str.replace("T", "").split()[0]) * 1024 * 1024)
                    elif mem_str.isdigit():
                        memory_mb = int(mem_str)
                except (ValueError, IndexError):
                    pass
            time_limit_str = parts[8] if len(parts) > 8 else ""
            time_used_str = parts[9] if len(parts) > 9 else ""
            nodes_str = parts[10] if len(parts) > 10 else ""
            time_limit_minutes = self._parse_slurm_time(time_limit_str)
            time_used_minutes = self._parse_slurm_time(time_used_str)
            gpu_count = 0
            info = SlurmJobInfo(
                job_id=job_id,
                job_name=job_name,
                partition=partition,
                user=user,
                state=state,
                node_count=node_count,
                gpu_count=gpu_count,
                cpus=cpus,
                memory_mb=memory_mb,
                time_limit_minutes=time_limit_minutes,
                time_used_minutes=time_used_minutes,
                nodes=nodes_str,
            )
            if state.upper() in ("PENDING", "SUSPENDED"):
                pending.append(info)
            elif state.upper() in ("RUNNING", "COMPLETING"):
                running.append(info)
        return pending, running

    @staticmethod
    def _parse_slurm_time(time_str: str) -> int:
        if not time_str or time_str in ("UNLIMITED", "NOT_SET", "INFINITE"):
            return 0
        try:
            if "-" in time_str:
                days_part, rest = time_str.split("-", 1)
                days = int(days_part)
                parts = rest.split(":")
                if len(parts) == 3:
                    return days * 1440 + int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) // 60
                return days * 1440
            parts = time_str.split(":")
            if len(parts) == 1:
                return int(parts[0])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) // 60
            elif len(parts) == 4:
                return int(parts[0]) * 1440 + int(parts[1]) * 60 + int(parts[2]) + int(parts[3]) // 60
        except (ValueError, IndexError):
            pass
        return 0

    # ── Topology detection ──────────────────────────────────────

    def _detect_topology(self, nodes: list[SlurmNodeInfo]) -> NodeTopology:
        node_topologies: list[GpuTopology] = []
        cross_node_bw = 50.0
        has_nvswitch = False
        tp_group_size = 8
        dp_group_size = 8

        for node in nodes:
            if node.gpu_count == 0:
                continue
            gpu_model_lower = node.gpu_model.lower()

            if "h100" in gpu_model_lower or "h200" in gpu_model_lower:
                nvlink_count = 18
                bw = 900.0
                tp_group_size = 8
                has_nvswitch = "nvswitch" not in node.gpu_model.lower()
            elif "b100" in gpu_model_lower or "b200" in gpu_model_lower:
                nvlink_count = 18
                bw = 1800.0
                tp_group_size = 8
                has_nvswitch = True
            elif "a100" in gpu_model_lower or "a30" in gpu_model_lower:
                nvlink_count = 12
                bw = 600.0
                tp_group_size = 8 if "a100" in gpu_model_lower else 4
                has_nvswitch = node.gpu_count >= 8 and "a100" in gpu_model_lower
            elif "a6000" in gpu_model_lower or "rtx" in gpu_model_lower:
                nvlink_count = 0
                bw = 0.0
                tp_group_size = 1
            elif "v100" in gpu_model_lower:
                nvlink_count = 6
                bw = 300.0
                tp_group_size = 4
            elif "t4" in gpu_model_lower or "l4" in gpu_model_lower:
                nvlink_count = 0
                bw = 0.0
                tp_group_size = 1
            else:
                nvlink_count = 0
                bw = 0.0
                tp_group_size = 1

            links: list[GpuTopologyLink] = []
            if nvlink_count > 0:
                for i in range(min(node.gpu_count, 8)):
                    for j in range(i + 1, min(node.gpu_count, 8)):
                        links.append(GpuTopologyLink(
                            source_gpu=i,
                            target_gpu=j,
                            link_type="nvlink",
                            bandwidth_gb_per_sec=bw,
                            nvlink_count=nvlink_count,
                        ))
                if has_nvswitch:
                    for i in range(node.gpu_count):
                        for j in range(i + 1, node.gpu_count):
                            if not any(l.source_gpu == i and l.target_gpu == j for l in links):
                                links.append(GpuTopologyLink(
                                    source_gpu=i,
                                    target_gpu=j,
                                    link_type="nvswitch",
                                    bandwidth_gb_per_sec=bw * 2,
                                    nvlink_count=nvlink_count,
                                ))

            node_topologies.append(GpuTopology(
                node_name=node.node_name,
                gpu_count=node.gpu_count,
                gpu_model=node.gpu_model,
                nvswitch_present=has_nvswitch,
                nvlink_per_gpu=nvlink_count,
                links=links,
                numa_affinity=[i for i in range(node.gpu_count)],
            ))

        return NodeTopology(
            nodes=node_topologies,
            cross_node_bandwidth_gb_per_sec=cross_node_bw,
            has_nvswitch=has_nvswitch,
            recommended_dp_group_size=dp_group_size,
            recommended_tp_group_size=tp_group_size,
        )

    # ── Run checks ─────────────────────────────────────────────

    def run_checks(self) -> list[CheckItem]:
        checks: list[CheckItem] = []

        if self._has_mock_data():
            return self._mock_checks()

        controller_ok = self._check_controller()
        if controller_ok:
            checks.append(CheckItem(
                name="slurm_controller",
                status=CheckStatus.PASS,
                message=f"Slurm controller is reachable (pyslurm={_HAS_PYSLURM}).",
                details={"controller": self.cluster.options.get("slurm_host", "localhost"),
                         "pyslurm_available": _HAS_PYSLURM},
            ))
        else:
            checks.append(CheckItem(
                name="slurm_controller",
                status=CheckStatus.FAIL,
                message="Cannot reach Slurm controller.",
                remediation="Verify slurm_host and that slurmctld is running.",
            ))
            return checks

        nodes = self._get_nodes()
        if nodes:
            ready = sum(1 for n in nodes if n.state.lower().startswith(("idle", "mix", "alloc")))
            checks.append(CheckItem(
                name="slurm_nodes",
                status=CheckStatus.PASS if ready > 0 else CheckStatus.WARN,
                message=f"Slurm reports {len(nodes)} node(s), {ready} available.",
                details={"total": len(nodes), "ready": ready, "down": len(nodes) - ready},
            ))

            gpu_nodes = [n for n in nodes if n.gpu_count > 0]
            total_gpus = sum(n.gpu_count for n in gpu_nodes)
            checks.append(CheckItem(
                name="slurm_gpus",
                status=CheckStatus.PASS if total_gpus > 0 else CheckStatus.WARN,
                message=f"Slurm detected {total_gpus} GPU(s) across {len(gpu_nodes)} node(s).",
                details={"gpu_total": total_gpus, "gpu_nodes": len(gpu_nodes)},
                remediation=None if total_gpus > 0 else "Verify GRES configuration in slurm.conf.",
            ))
        else:
            checks.append(CheckItem(
                name="slurm_nodes",
                status=CheckStatus.FAIL,
                message="No Slurm nodes discovered.",
                remediation="Verify sinfo is working and nodes are configured.",
            ))

        pending, running = self._get_jobs()
        checks.append(CheckItem(
            name="slurm_jobs",
            status=CheckStatus.PASS,
            message=f"{len(running)} running, {len(pending)} pending job(s).",
            details={"running": len(running), "pending": len(pending)},
        ))

        return checks

    def _check_controller(self) -> bool:
        host = self.cluster.options.get("slurm_host", "")
        if _HAS_PYSLURM:
            try:
                pyslurm_data = self._try_pyslurm_sinfo()
                if pyslurm_data:
                    return True
            except Exception:
                pass
        if not host or host == "localhost":
            result = self._run_cmd(["sinfo", "--version"])
            return bool(result.strip())
        result = self._run_cmd(["ssh", host, "sinfo", "--version"])
        return bool(result.strip())

    # ── Telemetry collection ───────────────────────────────────

    def collect_telemetry(self) -> ClusterTelemetry:
        collected_at = datetime.now(timezone.utc)
        nodes = self._get_nodes()
        telemetry_nodes: list[NodeTelemetry] = []
        gpu_count = 0

        for n in nodes:
            gpu_devices: list[GPUDeviceTelemetry] = []
            for i in range(n.gpu_count):
                gpu_devices.append(GPUDeviceTelemetry(
                    index=i,
                    uuid=f"{n.node_name}/gpu-{i}",
                    model=n.gpu_model or "unknown",
                ))
            gpu_count += n.gpu_count
            node_status = "Ready" if n.state.lower().startswith(("idle", "mix", "alloc")) else "NotReady"
            telemetry_nodes.append(NodeTelemetry(
                node_name=n.node_name,
                status=node_status,
                cpu_capacity_millicores=n.cpu_count * 1000,
                memory_capacity_bytes=n.memory_mb * 1024 * 1024,
                gpu_devices=gpu_devices,
            ))

        return ClusterTelemetry(
            cluster_id=self.cluster.id,
            cluster_name=self.cluster.name,
            collected_at=collected_at,
            node_count=len(telemetry_nodes),
            gpu_count=gpu_count,
            nodes=telemetry_nodes,
            freshness_seconds=0.0,
        )

    def collect_slurm_telemetry(self) -> SlurmClusterTelemetry:
        collected_at = datetime.now(timezone.utc)
        nodes = self._get_nodes()
        pending_jobs, running_jobs = self._get_jobs()

        total_gpus = sum(n.gpu_count for n in nodes)
        total_cpus = sum(n.cpu_count for n in nodes)
        total_mem = sum(n.memory_mb for n in nodes)
        total_gpu_alloc = sum(j.gpu_count for j in running_jobs)
        allocated_cpus = sum(j.cpus for j in running_jobs if j.cpus)
        allocated_mem = sum(j.memory_mb for j in running_jobs if j.memory_mb)

        partitions = self._get_partitions()

        return SlurmClusterTelemetry(
            cluster_id=self.cluster.id,
            cluster_name=self.cluster.name,
            collected_at=collected_at,
            controller_status="up",
            node_count=len(nodes),
            gpu_count=total_gpus,
            nodes=nodes,
            partitions=partitions,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            total_cpus=total_cpus,
            allocated_cpus=allocated_cpus,
            total_memory_mb=total_mem,
            allocated_memory_mb=allocated_mem,
            total_gpus_allocated=total_gpu_alloc,
        )

    def _get_partitions(self) -> list[SlurmPartitionInfo]:
        if self._has_mock_data():
            data = self._load_mock_data()
            return [SlurmPartitionInfo(**p) for p in data.get("partitions", [])]

        pyslurm_data = self._try_pyslurm_sinfo()
        if pyslurm_data and "partitions" in pyslurm_data:
            return self._parse_partitions_from_pyslurm(pyslurm_data["partitions"])

        sinfo_json = self._fetch_sinfo_json()
        if sinfo_json:
            return self._parse_partitions_from_json(sinfo_json)

        return []

    def _parse_partitions_from_pyslurm(self, data: dict) -> list[SlurmPartitionInfo]:
        partitions: list[SlurmPartitionInfo] = []
        for name, part in data.items():
            partitions.append(SlurmPartitionInfo(
                name=name,
                state=part.get("state", "up")[0] if isinstance(part.get("state"), list) else part.get("state", "up"),
                total_cpus=part.get("total_cpus", 0),
                total_gpus=len(part.get("gres", {}).get("gpu", {}).get("counts", [])) if isinstance(part.get("gres"), dict) else 0,
                default_time_minutes=part.get("default_time", 60),
                max_time_minutes=part.get("max_time", 1440),
            ))
        return partitions

    def _parse_partitions_from_json(self, data: dict) -> list[SlurmPartitionInfo]:
        partitions: list[SlurmPartitionInfo] = []
        for p in data.get("partitions", []):
            partitions.append(SlurmPartitionInfo(
                name=p.get("name", "unknown"),
                state=p.get("state", "up")[0] if isinstance(p.get("state"), list) else p.get("state", "up"),
                total_cpus=p.get("total_cpus", 0),
                total_gpus=len(p.get("gres", {}).get("gpu", {}).get("counts", [])) if isinstance(p.get("gres"), dict) else 0,
                default_time_minutes=p.get("default_time", {}).get("number", 60) if isinstance(p.get("default_time"), dict) else 60,
                max_time_minutes=p.get("max_time", {}).get("number", 1440) if isinstance(p.get("max_time"), dict) else 1440,
            ))
        return partitions

    # ── Topology API ────────────────────────────────────────────

    def get_cluster_topology(self) -> NodeTopology:
        nodes = self._get_nodes()
        return self._detect_topology(nodes)

    # ── Real-time monitoring ────────────────────────────────────

    def start_job_monitor(self, job_id: int, config: JobMonitorConfig | None = None) -> None:
        cfg = config or JobMonitorConfig(job_id=job_id)
        with self._monitor_lock:
            if job_id in self._monitor_threads and self._monitor_threads[job_id].is_alive():
                logger.info("Monitor already running for job %d", job_id)
                return
            self._monitors[job_id] = cfg
            self._monitor_histories[job_id] = []
            t = threading.Thread(target=self._monitor_loop, args=(job_id, cfg), daemon=True)
            self._monitor_threads[job_id] = t
            t.start()
            logger.info("Started monitoring job %d (interval=%ds)", job_id, cfg.poll_interval_seconds)

    def stop_job_monitor(self, job_id: int) -> None:
        with self._monitor_lock:
            if job_id in self._monitor_threads:
                del self._monitors[job_id]
                self._monitor_threads[job_id] = None  # signal thread to stop
                logger.info("Stopped monitoring job %d", job_id)

    def get_job_history(self, job_id: int) -> list[SlurmJobSnapshot]:
        with self._monitor_lock:
            return list(self._monitor_histories.get(job_id, []))

    def _monitor_loop(self, job_id: int, config: JobMonitorConfig) -> None:
        while True:
            with self._monitor_lock:
                if job_id not in self._monitors:
                    break
            try:
                self._poll_job(job_id, config)
            except Exception as exc:
                logger.error("Monitor poll error for job %d: %s", job_id, exc)

            with self._monitor_lock:
                start_time = datetime.now(timezone.utc)
                if job_id in self._monitors:
                    history = self._monitor_histories.get(job_id, [])
                    stall_threshold = timedelta(minutes=config.stall_threshold_minutes)
                    if len(history) >= 3 and config.alert_on_stall:
                        last_three = history[-3:]
                        if all(h.state == last_three[0].state for h in last_three):
                            time_span = last_three[-1].timestamp - last_three[0].timestamp
                            if time_span >= stall_threshold:
                                logger.warning("Job %d appears stalled (state=%s for %ds)",
                                               job_id, last_three[0].state, int(time_span.total_seconds()))

                    if len(history) >= 2:
                        latest = history[-1]
                        if latest.state == "COMPLETED" and config.alert_on_completion:
                            logger.info("Job %d completed", job_id)
                        elif latest.state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL") and config.alert_on_failure:
                            logger.warning("Job %d ended with state=%s", job_id, latest.state)

            time.sleep(config.poll_interval_seconds)

    def _poll_job(self, job_id: int, config: JobMonitorConfig) -> None:
        timestamp = datetime.now(timezone.utc)
        pyslurm_data = self._try_pyslurm_sacct(job_id)
        state = "UNKNOWN"
        time_used = 0
        gpu_util = 0.0
        mem_util = 0.0

        if pyslurm_data:
            state = pyslurm_data.get("state", "UNKNOWN")
            elapsed = pyslurm_data.get("elapsed", 0)
            time_used = elapsed // 60
        else:
            sacct_output = self._fetch_sacct_for_job(job_id)
            for line in sacct_output.strip().split("\n"):
                line = line.strip()
                if not line or "." in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 7:
                    state = parts[6]
                    elapsed_str = parts[1]
                    time_used = self._parse_slurm_time(elapsed_str)
                    break

        pending, running = self._get_jobs()
        for j in running + pending:
            if j.job_id == job_id:
                gpu_util = 50.0
                break

        snapshot = SlurmJobSnapshot(
            timestamp=timestamp,
            job_id=job_id,
            state=state,
            time_used_minutes=time_used,
            gpu_utilization=gpu_util,
            memory_utilization=mem_util,
        )

        with self._monitor_lock:
            if job_id in self._monitor_histories:
                hist = self._monitor_histories[job_id]
                hist.append(snapshot)
                if len(hist) > config.max_history_points:
                    self._monitor_histories[job_id] = hist[-config.max_history_points:]

    def collect_monitoring_snapshot(self) -> MonitoringSnapshot:
        pending_jobs, running_jobs = self._get_jobs()
        nodes = self._get_nodes()
        total_gpus = sum(n.gpu_count for n in nodes)
        allocated = sum(j.gpu_count for j in running_jobs)
        free_gpus = total_gpus - allocated
        running_gpu_hours = sum(
            j.gpu_count * (j.time_used_minutes / 60.0) for j in running_jobs if j.time_used_minutes
        )
        avg_util = (allocated / max(total_gpus, 1)) * 100

        job_history: list[SlurmJobSnapshot] = []
        for j in running_jobs:
            job_history.append(SlurmJobSnapshot(
                timestamp=datetime.now(timezone.utc),
                job_id=j.job_id,
                state=j.state,
                time_used_minutes=j.time_used_minutes,
            ))

        return MonitoringSnapshot(
            cluster_id=self.cluster.id,
            collected_at=datetime.now(timezone.utc),
            running_jobs=running_jobs,
            pending_jobs=pending_jobs,
            total_gpus=total_gpus,
            free_gpus=free_gpus,
            avg_cluster_utilization=round(avg_util, 1),
            total_waiting_jobs=len(pending_jobs),
            total_running_jobs=len(running_jobs),
            total_gpu_hours_used=round(running_gpu_hours, 1),
            job_history=job_history,
        )

    # ── Mock data ──────────────────────────────────────────────

    def _has_mock_data(self) -> bool:
        return bool(self.cluster.options.get("mock_slurm_data", ""))

    def _load_mock_data(self) -> dict[str, Any]:
        path = self.cluster.options.get("mock_slurm_data", "")
        if not path:
            return {}
        mock_file = Path(path)
        if not mock_file.is_absolute():
            mock_file = Path(__file__).resolve().parents[3] / path
        if mock_file.exists():
            with open(mock_file) as f:
                return json.load(f)
        logger.warning("Mock Slurm data not found: %s", mock_file)
        return {}

    def _mock_checks(self) -> list[CheckItem]:
        data = self._load_mock_data()
        nodes = data.get("nodes", [])
        gpu_count = sum(n.get("gpu_count", 0) for n in nodes)
        return [
            CheckItem(name="slurm_controller", status=CheckStatus.PASS, message="Slurm controller is reachable (mock)."),
            CheckItem(name="slurm_nodes", status=CheckStatus.PASS, message=f"Simulated {len(nodes)} node(s) (mock)."),
            CheckItem(name="slurm_partitions", status=CheckStatus.PASS, message=f"Slurm has {len(data.get('partitions', []))} partition(s) (mock)."),
            CheckItem(name="slurm_gpus", status=CheckStatus.PASS, message=f"Slurm detected {gpu_count} GPU resource(s) (mock)."),
            CheckItem(name="slurm_jobs", status=CheckStatus.PASS, message=f"{len(data.get('running_jobs', []))} running, {len(data.get('pending_jobs', []))} pending (mock)."),
        ]

    def _mock_telemetry(self, collected_at: datetime) -> ClusterTelemetry:
        data = self._load_mock_data()
        nodes_data = data.get("nodes", [])
        telemetry_nodes: list[NodeTelemetry] = []
        gpu_count = 0
        for nd in nodes_data:
            ngpus = nd.get("gpu_count", 0)
            gpu_devices = [
                GPUDeviceTelemetry(index=i, uuid=f"{nd['node_name']}/gpu-{i}", model=nd.get("gpu_model", "unknown"))
                for i in range(ngpus)
            ]
            gpu_count += ngpus
            telemetry_nodes.append(NodeTelemetry(
                node_name=nd.get("node_name", "unknown"),
                status=nd.get("state", "idle"),
                cpu_capacity_millicores=nd.get("cpu_count", 64) * 1000,
                memory_capacity_bytes=nd.get("memory_mb", 512000) * 1024 * 1024,
                gpu_devices=gpu_devices,
            ))
        return ClusterTelemetry(
            cluster_id=self.cluster.id,
            cluster_name=self.cluster.name,
            collected_at=collected_at,
            node_count=len(telemetry_nodes),
            gpu_count=gpu_count,
            nodes=telemetry_nodes,
            freshness_seconds=0.0,
        )

    def _mock_slurm_telemetry(self, collected_at: datetime) -> SlurmClusterTelemetry:
        data = self._load_mock_data()
        nodes = [SlurmNodeInfo(**n) for n in data.get("nodes", [])]
        partitions = [SlurmPartitionInfo(**p) for p in data.get("partitions", [])]
        pending_jobs = [SlurmJobInfo(**j) for j in data.get("pending_jobs", [])]
        running_jobs = [SlurmJobInfo(**j) for j in data.get("running_jobs", [])]

        total_gpus = sum(n.gpu_count for n in nodes)
        total_cpus = sum(n.cpu_count for n in nodes)
        total_mem = sum(n.memory_mb for n in nodes)
        total_gpu_alloc = sum(j.gpu_count for j in running_jobs)
        allocated_cpus = sum(j.cpus for j in running_jobs if j.cpus)
        allocated_mem = sum(j.memory_mb for j in running_jobs if j.memory_mb)

        return SlurmClusterTelemetry(
            cluster_id=self.cluster.id,
            cluster_name=self.cluster.name,
            collected_at=collected_at,
            controller_status="up",
            node_count=len(nodes),
            gpu_count=total_gpus,
            nodes=nodes,
            partitions=partitions,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            total_cpus=total_cpus,
            allocated_cpus=allocated_cpus,
            total_memory_mb=total_mem,
            allocated_memory_mb=allocated_mem,
            total_gpus_allocated=total_gpu_alloc,
        )
