from __future__ import annotations

from abc import ABC, abstractmethod

from gpuopt.schemas import CheckItem, ClusterRecord, ClusterTelemetry


class ClusterConnector(ABC):
    def __init__(self, cluster: ClusterRecord) -> None:
        self.cluster = cluster

    @abstractmethod
    def run_checks(self) -> list[CheckItem]:
        """Run read-only environment checks and return individual results."""

    def collect_telemetry(self) -> ClusterTelemetry:
        """Collect cluster telemetry and return normalized data.

        The base implementation returns an empty telemetry object.
        Connectors override this to provide real or mock telemetry.
        """
        return ClusterTelemetry(
            cluster_id=self.cluster.id,
            cluster_name=self.cluster.name,
        )
