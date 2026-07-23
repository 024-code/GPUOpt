from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from gpuopt.inference_schemas import GpuAllocation, GpuUsageResponse

logger = logging.getLogger(__name__)

GPU_MEMORY_MAP: dict[str, float] = {
    "h200": 141.0, "h100": 80.0, "b200": 180.0, "b100": 80.0,
    "a100": 80.0, "a6000": 48.0, "a5000": 24.0, "a40": 48.0,
    "a30": 24.0, "v100": 32.0, "v100s": 32.0, "t4": 16.0,
    "l40s": 48.0, "l4": 24.0,
    "rtx 4090": 24.0, "rtx 6000 ada": 48.0, "rtx a6000": 48.0,
}


class GpuInventoryService:

    def __init__(self) -> None:
        self._data_dir = Path(os.environ.get("GPUOPT_DATA_DIR", "/data")) / "inference"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _inventory_path(self, cluster_id: UUID) -> Path:
        return self._data_dir / f"gpu_inventory_{cluster_id}.json"

    def get_usage(self, cluster_id: UUID, cluster_name: str = "") -> GpuUsageResponse:
        path = self._inventory_path(cluster_id)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return GpuUsageResponse(**data)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load inventory for %s: %s", cluster_id, exc)

        default_response = GpuUsageResponse(
            cluster_id=cluster_id,
            cluster_name=cluster_name or f"cluster-{cluster_id}",
            timestamp=datetime.now(timezone.utc),
            total_gpus=0,
            allocated_gpus=0,
            available_gpus=0,
            utilization_pct=0.0,
            by_model=[],
            by_node=[],
        )
        self._save_usage(path, default_response)
        return default_response

    def update_usage(
        self,
        cluster_id: UUID,
        cluster_name: str,
        gpus_by_model: list[dict[str, Any]],
        gpus_by_node: list[dict[str, Any]],
    ) -> GpuUsageResponse:
        total = 0
        allocated = 0
        by_model: list[GpuAllocation] = []

        for item in gpus_by_model:
            gpu_model = item.get("gpu_model", "unknown")
            gpu_total = item.get("total", 0)
            gpu_allocated = item.get("allocated", 0)
            gpu_reserved = item.get("reserved", 0)
            gpu_available = gpu_total - gpu_allocated - gpu_reserved
            mem = GPU_MEMORY_MAP.get(gpu_model.lower(), 80.0)
            avg_util = item.get("average_utilization", 0.0)

            by_model.append(GpuAllocation(
                gpu_model=gpu_model,
                total=gpu_total,
                allocated=gpu_allocated,
                available=gpu_available,
                reserved=gpu_reserved,
                gpu_memory_gib=mem,
                average_utilization=avg_util,
            ))
            total += gpu_total
            allocated += gpu_allocated

        available = total - allocated
        util_pct = round((allocated / max(total, 1)) * 100, 1)

        response = GpuUsageResponse(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            timestamp=datetime.now(timezone.utc),
            total_gpus=total,
            allocated_gpus=allocated,
            available_gpus=available,
            utilization_pct=util_pct,
            by_model=by_model,
            by_node=gpus_by_node,
        )

        path = self._inventory_path(cluster_id)
        self._save_usage(path, response)
        return response

    def _save_usage(self, path: Path, response: GpuUsageResponse) -> None:
        path.write_text(json.dumps(response.model_dump(mode="json"), indent=2, default=str))
