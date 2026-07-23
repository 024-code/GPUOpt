from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


@dataclass
class WatchEvent:
    event_type: str
    resource_type: str
    resource_id: str
    resource_version: int
    data: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResourceCache:
    def __init__(self, max_versions: int = 50) -> None:
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self._max_versions = max_versions
        self._current_version: int = 0

    def set(self, resource_type: str, resource_id: str, data: dict[str, Any]) -> WatchEvent:
        with self._lock:
            self._current_version += 1
            key = f"{resource_type}:{resource_id}"
            self._cache[key] = data
            if key not in self._versions:
                self._versions[key] = []
            self._versions[key].append((self._current_version, dict(data)))
            if len(self._versions[key]) > self._max_versions:
                self._versions[key] = self._versions[key][-self._max_versions:]
            return WatchEvent(
                event_type="UPDATED",
                resource_type=resource_type,
                resource_id=resource_id,
                resource_version=self._current_version,
                data=data,
            )

    def delete(self, resource_type: str, resource_id: str) -> WatchEvent | None:
        with self._lock:
            key = f"{resource_type}:{resource_id}"
            if key not in self._cache:
                return None
            self._current_version += 1
            del self._cache[key]
            return WatchEvent(
                event_type="DELETED",
                resource_type=resource_type,
                resource_id=resource_id,
                resource_version=self._current_version,
                data={},
            )

    def get(self, resource_type: str, resource_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._cache.get(f"{resource_type}:{resource_id}")

    def list(self, resource_type: str) -> list[dict[str, Any]]:
        with self._lock:
            prefix = f"{resource_type}:"
            return [v for k, v in self._cache.items() if k.startswith(prefix)]

    def list_all(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            result: dict[str, list[dict[str, Any]]] = {}
            for key, value in self._cache.items():
                rtype = key.split(":")[0]
                if rtype not in result:
                    result[rtype] = []
                result[rtype].append(value)
            return result

    def get_resource_version(self, resource_type: str, resource_id: str) -> int:
        with self._lock:
            key = f"{resource_type}:{resource_id}"
            versions = self._versions.get(key, [])
            return versions[-1][0] if versions else 0

    def get_current_version(self) -> int:
        with self._lock:
            return self._current_version

    def get_changes_since(self, resource_type: str, since_version: int) -> list[WatchEvent]:
        with self._lock:
            events: list[WatchEvent] = []
            prefix = f"{resource_type}:"
            for key, versions in self._versions.items():
                if not key.startswith(prefix):
                    continue
                resource_id = key[len(prefix):]
                for ver, data in versions:
                    if ver > since_version:
                        events.append(WatchEvent(
                            event_type="UPDATED",
                            resource_type=resource_type,
                            resource_id=resource_id,
                            resource_version=ver,
                            data=data,
                        ))
            return sorted(events, key=lambda e: e.resource_version)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._versions.clear()
            self._current_version = 0


class WatchManager:
    def __init__(self, cache: ResourceCache | None = None) -> None:
        self._cache = cache or ResourceCache()
        self._watch_clients: dict[str, list[Callable]] = {}
        self._lock = threading.RLock()
        self._resync_interval = 300
        self._resync_timer: threading.Thread | None = None
        self._running = False

    def watch(self, resource_type: str, resource_version: int = 0,
              timeout_seconds: int = 30) -> list[WatchEvent]:
        if resource_version == 0:
            items = self._cache.list(resource_type)
            return [
                WatchEvent(event_type="INITIAL", resource_type=resource_type,
                           resource_id=item.get("id", ""), resource_version=0, data=item)
                for item in items
            ]
        return self._cache.get_changes_since(resource_type, resource_version)

    def watch_all(self, resource_version: int = 0) -> list[WatchEvent]:
        if resource_version == 0:
            events: list[WatchEvent] = []
            for rtype, items in self._cache.list_all().items():
                for item in items:
                    events.append(WatchEvent(
                        event_type="INITIAL", resource_type=rtype,
                        resource_id=item.get("id", ""), resource_version=0, data=item,
                    ))
            return events
        events = []
        for rtype in self._cache.list_all():
            events.extend(self._cache.get_changes_since(rtype, resource_version))
        return sorted(events, key=lambda e: e.resource_version)

    def subscribe(self, resource_type: str, callback: Callable) -> str:
        sub_id = str(uuid4())
        with self._lock:
            if resource_type not in self._watch_clients:
                self._watch_clients[resource_type] = []
            self._watch_clients[resource_type].append(callback)
        return sub_id

    def notify(self, event: WatchEvent) -> None:
        with self._lock:
            callbacks = list(self._watch_clients.get(event.resource_type, []))
            all_callbacks = list(self._watch_clients.get("*", []))
        for cb in callbacks + all_callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.error("Watch callback error: %s", exc)

    def resync(self, source_func: Callable[[], list[tuple[str, str, dict[str, Any]]]]) -> int:
        count = 0
        try:
            items = source_func()
            for rtype, rid, data in items:
                self._cache.set(rtype, rid, data)
                count += 1
            logger.info("Resync complete: %d items synced", count)
        except Exception as exc:
            logger.error("Resync failed: %s", exc)
        return count

    def start_auto_resync(self, source_func: Callable[[], list[tuple[str, str, dict[str, Any]]]],
                          interval_seconds: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._resync_interval = interval_seconds or self._resync_interval

        def _loop() -> None:
            while self._running:
                time.sleep(self._resync_interval)
                self.resync(source_func)

        self._resync_timer = threading.Thread(target=_loop, daemon=True)
        self._resync_timer.start()
        logger.info("Auto-resync started (interval=%ds)", self._resync_interval)

    def stop_auto_resync(self) -> None:
        self._running = False

    async def watch_stream(self, resource_type: str, since_version: int = 0) -> list[WatchEvent]:
        return await asyncio.to_thread(self.watch, resource_type, since_version)

    @property
    def cache(self) -> ResourceCache:
        return self._cache

    @property
    def current_version(self) -> int:
        return self._cache.get_current_version()


_default_watch_manager = WatchManager()


def get_watch_manager() -> WatchManager:
    return _default_watch_manager
