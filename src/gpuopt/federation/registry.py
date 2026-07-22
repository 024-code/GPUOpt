from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import ClusterHealth, FederatedCluster, FederationRole, FederationState

logger = logging.getLogger(__name__)


class FederatedClusterRegistry:
    def __init__(self) -> None:
        self._clusters: dict[str, FederatedCluster] = {}
        self._lock = threading.RLock()

    def register(self, name: str, endpoint: str = "", environment: str = "sandbox",
                 region: str = "", labels: dict[str, str] | None = None,
                 options: dict[str, Any] | None = None) -> FederatedCluster:
        with self._lock:
            for existing in self._clusters.values():
                if existing.name == name:
                    existing.endpoint = endpoint or existing.endpoint
                    existing.environment = environment or existing.environment
                    existing.region = region or existing.region
                    existing.last_seen = datetime.now(timezone.utc)
                    if labels:
                        existing.labels.update(labels)
                    if options:
                        existing.options.update(options)
                    logger.info("Federated cluster %s updated", name)
                    return existing

            cluster = FederatedCluster(
                name=name, endpoint=endpoint, environment=environment,
                region=region, labels=labels or {}, options=options or {},
            )
            self._clusters[cluster.id] = cluster
            logger.info("Federated cluster %s registered (id=%s)", name, cluster.id)
            return cluster

    def unregister(self, cluster_id: str) -> bool:
        with self._lock:
            if cluster_id in self._clusters:
                del self._clusters[cluster_id]
                logger.info("Federated cluster %s unregistered", cluster_id)
                return True
            return False

    def get(self, cluster_id: str) -> FederatedCluster | None:
        return self._clusters.get(cluster_id)

    def get_by_name(self, name: str) -> FederatedCluster | None:
        for c in self._clusters.values():
            if c.name == name:
                return c
        return None

    def list(self) -> list[FederatedCluster]:
        with self._lock:
            return list(self._clusters.values())

    def update_health(self, cluster_id: str, health: ClusterHealth, total_gpus: int = 0,
                      free_gpus: int = 0, gpu_models: list[str] | None = None,
                      avg_utilization: float = 0.0) -> FederatedCluster | None:
        with self._lock:
            cluster = self._clusters.get(cluster_id)
            if cluster is None:
                return None
            cluster.health = health
            cluster.total_gpus = total_gpus
            cluster.free_gpus = free_gpus
            if gpu_models is not None:
                cluster.gpu_models = gpu_models
            cluster.avg_utilization = avg_utilization
            cluster.last_seen = datetime.now(timezone.utc)
            return cluster

    def get_state(self) -> FederationState:
        with self._lock:
            clusters = list(self._clusters.values())
            return FederationState(
                clusters=clusters,
                total_gpus_across_clusters=sum(c.total_gpus for c in clusters),
                total_free_gpus=sum(c.free_gpus for c in clusters),
            )
