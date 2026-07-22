from __future__ import annotations

from gpuopt.schemas import ClusterRecord, ConnectorType

from .base import ClusterConnector
from .kubernetes import KubernetesConnector
from .mock import MockConnector
from .slurm import SlurmConnector


def build_connector(cluster: ClusterRecord) -> ClusterConnector:
    if cluster.connector_type == ConnectorType.MOCK:
        return MockConnector(cluster)
    if cluster.connector_type == ConnectorType.KUBERNETES:
        return KubernetesConnector(cluster)
    if cluster.connector_type == ConnectorType.SLURM:
        return SlurmConnector(cluster)
    raise ValueError(f"Unsupported connector type: {cluster.connector_type}")
