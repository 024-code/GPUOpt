from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from prometheus_client import Counter, Gauge, Histogram

from .connectors.factory import build_connector
from .ml.forecast_model import ForecastModel
from .repository import ClusterRepository
from .schemas import (
    CheckItem,
    CheckStatus,
    ClusterStateData,
    ClusterStateSummary,
    ClusterTelemetry,
    EnvironmentCheckReport,
    EnvironmentSummary,
    GPUDeviceState,
    NodeState,
)

logger = logging.getLogger(__name__)

CHECK_RUNS = Counter(
    "gpuopt_environment_check_runs_total",
    "Total Kubernetes environment check runs.",
    ["cluster", "environment", "status"],
)
CHECK_DURATION = Histogram(
    "gpuopt_environment_check_duration_seconds",
    "Environment check duration in seconds.",
    ["cluster", "environment"],
)
CLUSTER_STATUS = Gauge(
    "gpuopt_cluster_health_status",
    "Latest cluster status: 0 unchecked, 1 healthy, 2 warning, 3 failing.",
    ["cluster", "environment"],
)


class EnvironmentCheckService:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository

    def check_cluster(self, cluster_id: UUID) -> EnvironmentCheckReport:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        started_at = datetime.now(timezone.utc)
        connector = build_connector(cluster)
        with CHECK_DURATION.labels(cluster=cluster.name, environment=cluster.environment).time():
            try:
                checks = connector.run_checks()
            except Exception as exc:
                logger.exception("Environment check crashed for cluster %s", cluster.name)
                checks = [
                    CheckItem(
                        name="connector",
                        status=CheckStatus.FAIL,
                        message=f"Connector execution failed: {exc}",
                        remediation="Review connector configuration and logs.",
                    )
                ]
        completed_at = datetime.now(timezone.utc)
        overall = self._overall_status(checks)
        summary = {status.value: sum(item.status == status for item in checks) for status in CheckStatus}
        report = EnvironmentCheckReport(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            started_at=started_at,
            completed_at=completed_at,
            overall_status=overall,
            checks=checks,
            summary=summary,
        )
        self.repository.save_report(report)
        CHECK_RUNS.labels(
            cluster=cluster.name,
            environment=cluster.environment,
            status=overall.value,
        ).inc()
        CLUSTER_STATUS.labels(cluster=cluster.name, environment=cluster.environment).set(
            {CheckStatus.PASS: 1, CheckStatus.WARN: 2, CheckStatus.FAIL: 3, CheckStatus.SKIP: 0}[overall]
        )
        return report

    def check_all(self) -> list[EnvironmentCheckReport]:
        return [self.check_cluster(cluster.id) for cluster in self.repository.list_clusters()]

    def summarize(self) -> EnvironmentSummary:
        clusters = list(self.repository.latest_reports())
        counters = {"healthy": 0, "warning": 0, "failing": 0, "unchecked": 0}
        environments: dict[str, dict[str, int]] = {}
        for cluster, report in clusters:
            env = environments.setdefault(
                cluster.environment,
                {"clusters": 0, "healthy": 0, "warning": 0, "failing": 0, "unchecked": 0},
            )
            env["clusters"] += 1
            if report is None:
                key = "unchecked"
            elif report.overall_status == CheckStatus.PASS:
                key = "healthy"
            elif report.overall_status == CheckStatus.WARN:
                key = "warning"
            else:
                key = "failing"
            counters[key] += 1
            env[key] += 1
        return EnvironmentSummary(
            clusters=len(clusters),
            healthy=counters["healthy"],
            warning=counters["warning"],
            failing=counters["failing"],
            unchecked=counters["unchecked"],
            environments=environments,
        )

    @staticmethod
    def _overall_status(checks: list[CheckItem]) -> CheckStatus:
        if any(item.status == CheckStatus.FAIL for item in checks):
            return CheckStatus.FAIL
        if any(item.status == CheckStatus.WARN for item in checks):
            return CheckStatus.WARN
        if checks and all(item.status == CheckStatus.SKIP for item in checks):
            return CheckStatus.SKIP
        return CheckStatus.PASS


TELEMETRY_COLLECTIONS = Counter(
    "gpuopt_telemetry_collections_total",
    "Total telemetry collections by cluster.",
    ["cluster", "environment"],
)
STATE_FRESHNESS = Gauge(
    "gpuopt_state_freshness_seconds",
    "Age of the latest cluster state snapshot in seconds.",
    ["cluster", "environment"],
)
STATE_GPU_COUNT = Gauge(
    "gpuopt_state_gpu_count",
    "Number of GPUs in the latest cluster state.",
    ["cluster", "environment"],
)


class ClusterStateService:
    def __init__(self, repository: ClusterRepository, forecast_model: ForecastModel | None = None) -> None:
        self.repository = repository
        self.forecast_model = forecast_model or ForecastModel()

    def collect_state(self, cluster_id: UUID) -> ClusterStateData:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        connector = build_connector(cluster)

        telemetry: ClusterTelemetry = connector.collect_telemetry()
        collected_at = telemetry.collected_at

        nodes: list[NodeState] = []
        total_gpu_memory = 0
        for node_telemetry in telemetry.nodes:
            gpu_devices = [
                GPUDeviceState(
                    index=gpu.index,
                    uuid=gpu.uuid,
                    model=gpu.model,
                    memory_total_bytes=gpu.memory_total_bytes,
                    memory_used_bytes=gpu.memory_used_bytes,
                    status="healthy",
                )
                for gpu in node_telemetry.gpu_devices
            ]
            total_gpu_memory += sum(g.memory_total_bytes for g in gpu_devices)
            nodes.append(
                NodeState(
                    name=node_telemetry.node_name,
                    status=node_telemetry.status,
                    gpu_devices=gpu_devices,
                    pod_count=node_telemetry.pod_count,
                    pod_capacity=node_telemetry.pod_capacity,
                )
            )

        state = ClusterStateData(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            collected_at=collected_at,
            node_count=telemetry.node_count,
            gpu_count=telemetry.gpu_count,
            total_gpu_memory_bytes=total_gpu_memory,
            nodes=nodes,
            telemetry=telemetry,
        )
        self.repository.save_state(state)

        self.forecast_model.update_history(state)

        TELEMETRY_COLLECTIONS.labels(
            cluster=cluster.name, environment=cluster.environment
        ).inc()
        STATE_FRESHNESS.labels(
            cluster=cluster.name, environment=cluster.environment
        ).set(state.freshness_seconds)
        STATE_GPU_COUNT.labels(
            cluster=cluster.name, environment=cluster.environment
        ).set(telemetry.gpu_count)

        return state

    def get_latest_state(self, cluster_id: UUID) -> ClusterStateData | None:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.latest_state(cluster_id)

    def summarize_all(self) -> list[ClusterStateSummary]:
        summaries: list[ClusterStateSummary] = []
        now = datetime.now(timezone.utc)
        for cluster, state in self.repository.list_state_summaries():
            if state is None:
                summaries.append(
                    ClusterStateSummary(
                        cluster_id=cluster.id,
                        cluster_name=cluster.name,
                        environment=cluster.environment,
                        status="unchecked",
                    )
                )
            else:
                age = (now - state.collected_at).total_seconds()
                healthy_nodes = sum(1 for n in state.nodes if n.status == "Ready")
                status = "fresh" if age < 60 else ("recent" if age < 300 else "stale")
                summaries.append(
                    ClusterStateSummary(
                        cluster_id=cluster.id,
                        cluster_name=cluster.name,
                        environment=cluster.environment,
                        last_collected_at=state.collected_at,
                        age_seconds=round(age, 1),
                        node_count=state.node_count,
                        gpu_count=state.gpu_count,
                        healthy_nodes=healthy_nodes,
                        total_gpu_memory_bytes=state.total_gpu_memory_bytes,
                        status=status,
                    )
                )
        return summaries
