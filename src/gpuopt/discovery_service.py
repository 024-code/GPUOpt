from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from .hardware_detector import _run_remote

logger = logging.getLogger(__name__)


class NodeDiscoveryTarget:
    def __init__(
        self,
        host: str,
        ssh_key_file: str | None = None,
        agent_endpoint: str | None = None,
        node_id: str = "",
    ) -> None:
        self.host = host
        self.ssh_key_file = ssh_key_file
        self.agent_endpoint = agent_endpoint
        self.node_id = node_id or host


class DiscoveryResult:
    def __init__(
        self,
        node_id: str,
        host: str,
        success: bool,
        hardware: dict[str, Any] | None = None,
        error: str = "",
        timestamp: str = "",
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.success = success
        self.hardware = hardware or {}
        self.error = error
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "success": self.success,
            "hardware": self.hardware,
            "error": self.error,
            "timestamp": self.timestamp,
        }


def discover_via_ssh(target: NodeDiscoveryTarget) -> DiscoveryResult:
    try:
        agent_script = (
            "python3 -c \"import sys; sys.path.insert(0, '.'); "
            "from gpuopt.agent import run_agent_once; run_agent_once()\""
        )
        output = _run_remote(
            target.host,
            ["python3", "-c", (
                "import sys, json; "
                "sys.path.insert(0, '.'); "
                "from gpuopt.agent import detect_local_hardware; "
                "print(json.dumps(detect_local_hardware(), default=str))"
            )],
            timeout=30,
            key_file=target.ssh_key_file,
        )
        if not output:
            return DiscoveryResult(
                target.node_id, target.host, False,
                error="SSH returned no output",
            )
        hardware = json.loads(output)
        return DiscoveryResult(target.node_id, target.host, True, hardware=hardware)
    except Exception as exc:
        return DiscoveryResult(target.node_id, target.host, False, error=str(exc))


def discover_via_http(target: NodeDiscoveryTarget) -> DiscoveryResult:
    import urllib.request
    try:
        url = target.agent_endpoint.rstrip("/") + "/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return DiscoveryResult(
                    target.node_id, target.host, False,
                    error=f"HTTP {resp.status}",
                )
            body = resp.read().decode("utf-8")
            hardware = json.loads(body) if body else {}
        return DiscoveryResult(target.node_id, target.host, True, hardware=hardware)
    except Exception as exc:
        return DiscoveryResult(target.node_id, target.host, False, error=str(exc))


def discover_node(target: NodeDiscoveryTarget) -> DiscoveryResult:
    if target.agent_endpoint:
        return discover_via_http(target)
    return discover_via_ssh(target)


def discover_cluster(
    targets: list[NodeDiscoveryTarget],
    concurrency: int = 4,
) -> list[DiscoveryResult]:
    from concurrent.futures import ThreadPoolExecutor
    results: list[DiscoveryResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(discover_node, t) for t in targets]
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(DiscoveryResult("unknown", "unknown", False, error=str(exc)))
    return results


class AutoDiscoveryService:
    def __init__(
        self,
        targets: list[NodeDiscoveryTarget] | None = None,
        interval_seconds: int = 300,
        concurrency: int = 4,
        store_callback: Any | None = None,
    ) -> None:
        self.targets = targets or []
        self.interval_seconds = interval_seconds
        self.concurrency = concurrency
        self.store_callback = store_callback
        self._task: asyncio.Task | None = None
        self._stop = False

    def add_target(self, target: NodeDiscoveryTarget) -> None:
        self.targets.append(target)

    def remove_target(self, host: str) -> None:
        self.targets = [t for t in self.targets if t.host != host]

    def discover(self) -> list[DiscoveryResult]:
        return discover_cluster(self.targets, self.concurrency)

    async def discover_async(self) -> list[DiscoveryResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.discover)

    async def _run_loop(self) -> None:
        while not self._stop:
            logger.info("Auto-discovery running for %d targets", len(self.targets))
            try:
                results = await self.discover_async()
                healthy = sum(1 for r in results if r.success)
                logger.info("Discovered %d/%d nodes", healthy, len(results))
                if self.store_callback:
                    await self.store_callback(results)
            except Exception as exc:
                logger.error("Discovery cycle failed: %s", exc)
            for _ in range(self.interval_seconds):
                if self._stop:
                    break
                await asyncio.sleep(1)
        logger.info("Auto-discovery stopped")

    async def start(self) -> None:
        if self._task is None:
            self._stop = False
            self._task = asyncio.create_task(self._run_loop())
            logger.info("Auto-discovery service started")

    async def stop(self) -> None:
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Auto-discovery service stopped")
