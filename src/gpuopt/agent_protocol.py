from __future__ import annotations

import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    DECOMMISSIONED = "decommissioned"


class AgentEventType(str, Enum):
    REGISTERED = "agent.registered"
    HEARTBEAT = "agent.heartbeat"
    STATUS_CHANGE = "agent.status_change"
    TELEMETRY = "agent.telemetry"
    DISCONNECTED = "agent.disconnected"
    RECONNECTED = "agent.reconnected"
    ERROR = "agent.error"


@dataclass
class AgentRegistration:
    agent_id: str
    hostname: str
    version: str
    capabilities: list[str]
    labels: dict[str, str]
    registered_at: str
    last_heartbeat: str
    status: AgentStatus
    public_key_pem: str = ""
    mTLS_enabled: bool = False
    api_key_hash: str = ""


@dataclass
class AgentHeartbeat:
    agent_id: str
    timestamp: str
    sequence: int
    status: AgentStatus
    load: dict[str, float]
    metrics_summary: dict[str, Any]


@dataclass
class AgentStatusEvent:
    event_id: str
    agent_id: str
    event_type: AgentEventType
    timestamp: str
    previous_status: AgentStatus | None
    new_status: AgentStatus
    message: str
    details: dict[str, Any]


@dataclass
class AgentMTLSConfig:
    enabled: bool = False
    cert_file: str = ""
    key_file: str = ""
    ca_cert_file: str = ""
    verify_client: bool = True
    min_tls_version: str = "TLSv1.3"
    cert_revocation_check: bool = False


class AgentRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: dict[str, AgentRegistration] = {}
        self._heartbeats: dict[str, AgentHeartbeat] = {}
        self._events: list[AgentStatusEvent] = []
        self._event_limit = 10000
        self._heartbeat_timeout_seconds = 180
        self._check_timer: threading.Thread | None = None
        self._running = False

    def register(
        self,
        hostname: str,
        version: str,
        capabilities: list[str] | None = None,
        labels: dict[str, str] | None = None,
        public_key_pem: str = "",
        api_key_hash: str = "",
        mTLS_enabled: bool = False,
    ) -> AgentRegistration:
        agent_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        registration = AgentRegistration(
            agent_id=agent_id,
            hostname=hostname,
            version=version,
            capabilities=capabilities or [],
            labels=labels or {},
            registered_at=now,
            last_heartbeat=now,
            status=AgentStatus.REGISTERED,
            public_key_pem=public_key_pem,
            mTLS_enabled=mTLS_enabled,
            api_key_hash=api_key_hash,
        )
        with self._lock:
            self._agents[agent_id] = registration
            self._record_event(
                agent_id, AgentEventType.REGISTERED,
                None, AgentStatus.REGISTERED,
                f"Agent {hostname} v{version} registered",
                {"hostname": hostname, "version": version, "capabilities": capabilities},
            )
        logger.info("Agent registered: %s (%s)", agent_id, hostname)
        return registration

    def process_heartbeat(self, agent_id: str, load: dict[str, float] | None = None,
                          metrics_summary: dict[str, Any] | None = None) -> AgentHeartbeat | None:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                logger.warning("Heartbeat from unknown agent: %s", agent_id)
                return None
            now = datetime.now(timezone.utc)
            ts = now.isoformat()
            seq = self._heartbeats.get(agent_id).sequence + 1 if agent_id in self._heartbeats else 1
            agent.last_heartbeat = ts
            old_status = agent.status
            agent.status = AgentStatus.ACTIVE
            hb = AgentHeartbeat(
                agent_id=agent_id, timestamp=ts, sequence=seq,
                status=AgentStatus.ACTIVE,
                load=load or {}, metrics_summary=metrics_summary or {},
            )
            self._heartbeats[agent_id] = hb
            if old_status != AgentStatus.ACTIVE:
                self._record_event(
                    agent_id, AgentEventType.STATUS_CHANGE,
                    old_status, AgentStatus.ACTIVE,
                    f"Agent {agent.hostname} became active",
                    {"previous_status": old_status.value if old_status else None},
                )
            return hb

    def update_status(self, agent_id: str, new_status: AgentStatus, message: str = "",
                      details: dict[str, Any] | None = None) -> bool:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            old_status = agent.status
            if old_status == new_status:
                return True
            agent.status = new_status
            self._record_event(
                agent_id, AgentEventType.STATUS_CHANGE,
                old_status, new_status,
                message or f"Status changed: {old_status.value} -> {new_status.value}",
                details or {},
            )
            return True

    def check_stale_agents(self) -> list[str]:
        stale: list[str] = []
        now = datetime.now(timezone.utc)
        with self._lock:
            for agent_id, agent in list(self._agents.items()):
                if agent.status in (AgentStatus.DECOMMISSIONED, AgentStatus.UNREACHABLE):
                    continue
                last = datetime.fromisoformat(agent.last_heartbeat)
                elapsed = (now - last).total_seconds()
                if elapsed > self._heartbeat_timeout_seconds:
                    old_status = agent.status
                    agent.status = AgentStatus.UNREACHABLE
                    self._record_event(
                        agent_id, AgentEventType.STATUS_CHANGE,
                        old_status, AgentStatus.UNREACHABLE,
                        f"Agent {agent.hostname} unreachable for {elapsed:.0f}s",
                        {"elapsed_seconds": elapsed, "timeout": self._heartbeat_timeout_seconds},
                    )
                    stale.append(agent_id)
        return stale

    def start_stale_checker(self, interval_seconds: int = 60) -> None:
        if self._running:
            return
        self._running = True

        def _check_loop() -> None:
            while self._running:
                stale = self.check_stale_agents()
                if stale:
                    logger.warning("Stale agents detected: %s", stale)
                time.sleep(interval_seconds)

        self._check_timer = threading.Thread(target=_check_loop, daemon=True)
        self._check_timer.start()

    def stop_stale_checker(self) -> None:
        self._running = False

    def get_agent(self, agent_id: str) -> AgentRegistration | None:
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self, status: AgentStatus | None = None) -> list[AgentRegistration]:
        with self._lock:
            if status:
                return [a for a in self._agents.values() if a.status == status]
            return list(self._agents.values())

    def list_events(self, agent_id: str | None = None, limit: int = 100) -> list[AgentStatusEvent]:
        with self._lock:
            events = self._events
            if agent_id:
                events = [e for e in events if e.agent_id == agent_id]
            return events[-limit:]

    def get_latest_heartbeat(self, agent_id: str) -> AgentHeartbeat | None:
        with self._lock:
            return self._heartbeats.get(agent_id)

    def decommission(self, agent_id: str) -> bool:
        return self.update_status(agent_id, AgentStatus.DECOMMISSIONED, "Agent decommissioned")

    def _record_event(self, agent_id: str, event_type: AgentEventType,
                      previous: AgentStatus | None, new_status: AgentStatus,
                      message: str, details: dict[str, Any]) -> None:
        event = AgentStatusEvent(
            event_id=str(uuid4()),
            agent_id=agent_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            previous_status=previous,
            new_status=new_status,
            message=message,
            details=details,
        )
        self._events.append(event)
        if len(self._events) > self._event_limit:
            self._events = self._events[-self._event_limit:]

    def get_agent_count(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for a in self._agents.values():
                counts[a.status.value] = counts.get(a.status.value, 0) + 1
            return counts

    def get_heartbeat_stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._agents)
            active = sum(1 for a in self._agents.values() if a.status == AgentStatus.ACTIVE)
            stale = sum(1 for a in self._agents.values() if a.status == AgentStatus.UNREACHABLE)
            return {"total_agents": total, "active": active, "stale": stale, "registered": total - active - stale}


_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    return _registry


def create_mTLS_context(config: AgentMTLSConfig) -> ssl.SSLContext:
    ctx = ssl.create_default_context(
        purpose=ssl.Purpose.CLIENT_AUTH if config.verify_client else ssl.Purpose.SERVER_AUTH,
    )
    if config.cert_file and config.key_file:
        ctx.load_cert_chain(config.cert_file, config.key_file)
    if config.ca_cert_file:
        ctx.load_verify_locations(config.ca_cert_file)
    if config.verify_client:
        ctx.verify_mode = ssl.CERT_REQUIRED
    else:
        ctx.verify_mode = ssl.CERT_NONE
    if config.min_tls_version == "TLSv1.3":
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    else:
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx
