from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.params import Depends

from .dependencies import get_alert_manager, get_repository, get_watch_manager
from .s23_features import AlertManager
from .repository import ClusterRepository
from .watch_stream import WatchManager, WatchEvent

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._user_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str = "global", user: str = "") -> None:
        await websocket.accept()
        if channel not in self._connections:
            self._connections[channel] = set()
        self._connections[channel].add(websocket)
        if user:
            if user not in self._user_connections:
                self._user_connections[user] = set()
            self._user_connections[user].add(websocket)
        logger.info("WebSocket connected: channel=%s user=%s (%d total)", channel, user, len(self._connections.get(channel, set())))

    def disconnect(self, websocket: WebSocket, channel: str = "global", user: str = "") -> None:
        if channel in self._connections:
            self._connections[channel].discard(websocket)
            if not self._connections[channel]:
                del self._connections[channel]
        if user and user in self._user_connections:
            self._user_connections[user].discard(websocket)
            if not self._user_connections[user]:
                del self._user_connections[user]
        logger.info("WebSocket disconnected: channel=%s", channel)

    async def broadcast(self, channel: str, message: dict[str, Any]) -> int:
        sent = 0
        if channel not in self._connections:
            return sent
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        for ws in self._connections[channel]:
            try:
                await ws.send_text(payload)
                sent += 1
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections[channel].discard(ws)
        return sent

    async def send_to_user(self, user: str, message: dict[str, Any]) -> int:
        sent = 0
        if user not in self._user_connections:
            return sent
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        for ws in self._user_connections[user]:
            try:
                await ws.send_text(payload)
                sent += 1
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._user_connections[user].discard(ws)
        return sent

    @property
    def active_connections(self) -> int:
        return sum(len(ws) for ws in self._connections.values())


manager = ConnectionManager()


class StreamService:
    def __init__(self, repository: ClusterRepository, alert_manager: AlertManager | None = None) -> None:
        self.repository = repository
        self.alert_manager = alert_manager

    async def stream_cluster_state(self, websocket: WebSocket, cluster_id: UUID) -> None:
        await manager.connect(websocket, channel=f"cluster:{cluster_id}")
        try:
            while True:
                state = self.repository.latest_state(cluster_id)
                if state:
                    await websocket.send_json({
                        "type": "state_update",
                        "cluster_id": str(cluster_id),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": state.model_dump(mode="json"),
                    })
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now(timezone.utc).isoformat()})
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Stream error for cluster %s: %s", cluster_id, exc)
        finally:
            manager.disconnect(websocket, channel=f"cluster:{cluster_id}")

    async def stream_alerts(self, websocket: WebSocket, cluster_id: UUID | None = None) -> None:
        channel = f"alerts:{cluster_id}" if cluster_id else "alerts:all"
        await manager.connect(websocket, channel=channel)
        try:
            while True:
                alerts = self.alert_manager.list_alerts(cluster_id) if self.alert_manager else []
                firing = [a for a in alerts if a.status == "firing"]
                await websocket.send_json({
                    "type": "alert_update",
                    "cluster_id": str(cluster_id) if cluster_id else "all",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "firing_count": len(firing),
                    "alerts": [a.model_dump(mode="json") for a in firing[:50]],
                })
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Alert stream error: %s", exc)
        finally:
            manager.disconnect(websocket, channel=channel)

    async def stream_metrics(self, websocket: WebSocket, cluster_id: UUID) -> None:
        await manager.connect(websocket, channel=f"metrics:{cluster_id}")
        try:
            while True:
                state = self.repository.latest_state(cluster_id)
                metrics: dict[str, Any] = {
                    "type": "metrics_update",
                    "cluster_id": str(cluster_id),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "gpu_count": state.gpu_count if state else 0,
                    "node_count": state.node_count if state else 0,
                }
                if state and state.telemetry:
                    tele = state.telemetry
                    gpu_utils = [g.utilization_gpu_percent for n in tele.nodes for g in n.gpu_devices]
                    temps = [g.temperature_gpu_celsius for n in tele.nodes for g in n.gpu_devices if g.temperature_gpu_celsius > 0]
                    powers = [g.power_draw_watts for n in tele.nodes for g in n.gpu_devices]
                    metrics.update({
                        "avg_gpu_utilization": round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else 0,
                        "max_temperature": round(max(temps), 1) if temps else 0,
                        "total_power_watts": round(sum(powers), 1),
                    })
                await websocket.send_json(metrics)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Metrics stream error: %s", exc)
        finally:
            manager.disconnect(websocket, channel=f"metrics:{cluster_id}")


    async def watch_resource(
        self, websocket: WebSocket, resource_type: str,
        watch_manager: WatchManager | None = None,
    ) -> None:
        manager = watch_manager or get_watch_manager()
        await manager.cache  # ensure initialized
        await websocket.accept()
        current_version = 0
        try:
            events = await manager.watch_stream(resource_type, current_version)
            for event in events:
                await websocket.send_json({
                    "type": "watch_event",
                    "event_type": event.event_type,
                    "resource_type": event.resource_type,
                    "resource_id": event.resource_id,
                    "resource_version": event.resource_version,
                    "data": event.data,
                    "timestamp": event.timestamp,
                })
                if event.resource_version > current_version:
                    current_version = event.resource_version

            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                    msg = json.loads(data)
                    if msg.get("type") == "watch":
                        since = msg.get("since_version", current_version)
                        events = await manager.watch_stream(resource_type, since)
                        for event in events:
                            await websocket.send_json({
                                "type": "watch_event",
                                "event_type": event.event_type,
                                "resource_type": event.resource_type,
                                "resource_id": event.resource_id,
                                "resource_version": event.resource_version,
                                "data": event.data,
                                "timestamp": event.timestamp,
                            })
                            if event.resource_version > current_version:
                                current_version = event.resource_version
                    elif msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "current_version": current_version})
                except asyncio.TimeoutError:
                    await websocket.send_json({
                        "type": "keepalive",
                        "current_version": current_version,
                        "resource_type": resource_type,
                    })
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Watch stream error for %s: %s", resource_type, exc)


streaming_router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])


@streaming_router.websocket("/watch/{resource_type}")
async def ws_watch_resource(
    websocket: WebSocket,
    resource_type: str,
    watch_manager: WatchManager = Depends(get_watch_manager),
    repo: ClusterRepository = Depends(get_repository),
) -> None:
    svc = StreamService(repository=repo)
    await svc.watch_resource(websocket, resource_type, watch_manager)


@streaming_router.get("/watch/{resource_type}")
async def rest_watch_resource(
    resource_type: str,
    since_version: int = Query(0, ge=0),
    watch_manager: WatchManager = Depends(get_watch_manager),
) -> list[dict]:
    events = watch_manager.watch(resource_type, since_version)
    return [
        {
            "event_type": e.event_type,
            "resource_type": e.resource_type,
            "resource_id": e.resource_id,
            "resource_version": e.resource_version,
            "data": e.data,
            "timestamp": e.timestamp,
        }
        for e in events
    ]


@streaming_router.get("/watch")
async def rest_watch_all(
    since_version: int = Query(0, ge=0),
    watch_manager: WatchManager = Depends(get_watch_manager),
) -> list[dict]:
    events = watch_manager.watch_all(since_version)
    return [
        {
            "event_type": e.event_type,
            "resource_type": e.resource_type,
            "resource_id": e.resource_id,
            "resource_version": e.resource_version,
            "data": e.data,
            "timestamp": e.timestamp,
        }
        for e in events
    ]


@streaming_router.get("/cache/status")
async def cache_status(
    watch_manager: WatchManager = Depends(get_watch_manager),
) -> dict:
    return {
        "cache_size": watch_manager.cache.size(),
        "current_version": watch_manager.cache.get_current_version(),
        "resource_types": list(watch_manager.cache.list_all().keys()),
    }


@streaming_router.post("/cache/resync")
async def cache_resync(
    watch_manager: WatchManager = Depends(get_watch_manager),
    repo: ClusterRepository = Depends(get_repository),
) -> dict:
    def _source() -> list[tuple[str, str, dict[str, Any]]]:
        items: list[tuple[str, str, dict[str, Any]]] = []
        for cluster in repo.list_clusters():
            cid = str(cluster.id)
            items.append(("cluster", cid, {
                "id": cid, "name": cluster.name, "environment": cluster.environment,
                "connector_type": cluster.connector_type,
            }))
            state = repo.latest_state(cluster.id)
            if state:
                items.append(("state", cid, state.model_dump(mode="json")))
        return items

    synced = watch_manager.resync(_source)
    return {"synced": synced, "current_version": watch_manager.cache.get_current_version()}


@streaming_router.websocket("/cluster/{cluster_id}")
async def ws_cluster_state(
    websocket: WebSocket,
    cluster_id: str,
    repo: ClusterRepository = Depends(get_repository),
) -> None:
    from uuid import UUID
    svc = StreamService(repository=repo)
    await svc.stream_cluster_state(websocket, UUID(cluster_id))


@streaming_router.websocket("/alerts")
async def ws_alerts_all(
    websocket: WebSocket,
    alert_manager: AlertManager = Depends(get_alert_manager),
    repo: ClusterRepository = Depends(get_repository),
) -> None:
    svc = StreamService(repository=repo, alert_manager=alert_manager)
    await svc.stream_alerts(websocket)


@streaming_router.websocket("/alerts/{cluster_id}")
async def ws_alerts_cluster(
    websocket: WebSocket,
    cluster_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager),
    repo: ClusterRepository = Depends(get_repository),
) -> None:
    from uuid import UUID
    svc = StreamService(repository=repo, alert_manager=alert_manager)
    await svc.stream_alerts(websocket, UUID(cluster_id))


@streaming_router.websocket("/metrics/{cluster_id}")
async def ws_metrics(
    websocket: WebSocket,
    cluster_id: str,
    repo: ClusterRepository = Depends(get_repository),
) -> None:
    from uuid import UUID
    svc = StreamService(repository=repo)
    await svc.stream_metrics(websocket, UUID(cluster_id))
