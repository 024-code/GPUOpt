from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from .autoscaler import AutoscalerEngine, AutoscalingConfig, NodeGroupConfig, ScaleDirection, ScalingPolicy
from .gpu_monitor import GPUMonitor
from .preemption import PreemptionEngine, PreemptionPolicy, PreemptionPolicyConfig, PriorityClass

logger = logging.getLogger(__name__)

monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["gpu-monitoring", "preemption", "autoscaling"])

_gpu_monitor: GPUMonitor | None = None
_preemption_engine: PreemptionEngine | None = None
_autoscaler_engine: AutoscalerEngine | None = None


def _get_gpu_monitor() -> GPUMonitor:
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor


def _get_preemption_engine() -> PreemptionEngine:
    global _preemption_engine
    if _preemption_engine is None:
        _preemption_engine = PreemptionEngine()
    return _preemption_engine


def _get_autoscaler_engine() -> AutoscalerEngine:
    global _autoscaler_engine
    if _autoscaler_engine is None:
        _autoscaler_engine = AutoscalerEngine()
    return _autoscaler_engine


@monitoring_router.get("/gpu/snapshot")
def get_gpu_snapshot() -> dict:
    monitor = _get_gpu_monitor()
    snap = monitor.collect()
    return {
        "collected_at": snap.collected_at,
        "total_gpus": snap.total_gpus,
        "total_memory_mb": snap.total_memory_mb,
        "used_memory_mb": snap.used_memory_mb,
        "free_memory_mb": snap.free_memory_mb,
        "devices": [
            {
                "index": d.index,
                "uuid": d.uuid,
                "model": d.model,
                "memory_total_mb": d.memory_total_mb,
                "memory_used_mb": d.memory_used_mb,
                "memory_free_mb": d.memory_free_mb,
                "utilization_gpu_percent": d.utilization_gpu_percent,
                "utilization_memory_percent": d.utilization_memory_percent,
                "temperature_celsius": d.temperature_celsius,
                "power_draw_watts": d.power_draw_watts,
                "power_limit_watts": d.power_limit_watts,
                "fan_speed_percent": d.fan_speed_percent,
                "pcie_link_gen": d.pcie_link_gen,
                "pcie_link_width": d.pcie_link_width,
                "clock_sm_mhz": d.clock_sm_mhz,
                "clock_mem_mhz": d.clock_mem_mhz,
                "ecc_errors_volatile": d.ecc_errors_volatile,
                "ecc_errors_aggregate": d.ecc_errors_aggregate,
                "processes": [
                    {
                        "pid": p.pid,
                        "process_name": p.process_name,
                        "used_gpu_memory_mb": p.used_gpu_memory_mb,
                        "gpu_index": p.gpu_index,
                        "gpu_utilization": p.gpu_utilization,
                    }
                    for p in d.processes
                ],
            }
            for d in snap.devices
        ],
    }


@monitoring_router.get("/gpu/snapshot/cached")
def get_cached_gpu_snapshot() -> dict:
    monitor = _get_gpu_monitor()
    snap = monitor.get_snapshot()
    if snap is None:
        return {"status": "no_data", "message": "No snapshot collected yet. Start the monitor first."}
    return {
        "collected_at": snap.collected_at,
        "total_gpus": snap.total_gpus,
        "total_memory_mb": snap.total_memory_mb,
        "used_memory_mb": snap.used_memory_mb,
        "free_memory_mb": snap.free_memory_mb,
    }


@monitoring_router.post("/gpu/start")
def start_gpu_monitor(data: dict[str, Any]) -> dict:
    poll_interval = data.get("poll_interval", 15.0)
    monitor = _get_gpu_monitor()
    monitor._poll_interval = poll_interval
    monitor.start()
    return {"status": "started", "poll_interval": poll_interval}


@monitoring_router.post("/gpu/stop")
def stop_gpu_monitor() -> dict:
    _get_gpu_monitor().stop()
    return {"status": "stopped"}


@monitoring_router.get("/gpu/status")
def gpu_monitor_status() -> dict:
    monitor = _get_gpu_monitor()
    return {
        "running": monitor._running,
        "poll_interval": monitor._poll_interval,
        "has_snapshot": monitor._snapshot is not None,
    }


@monitoring_router.get("/preemption/history")
def get_preemption_history(limit: int = 50) -> list[dict]:
    engine = _get_preemption_engine()
    return [
        {
            "workload_name": a.workload_name,
            "namespace": a.namespace,
            "priority": a.priority.value,
            "reason": a.reason,
            "eviction_strategy": a.eviction_strategy,
            "initiated_at": a.initiated_at,
            "status": a.status,
        }
        for a in engine.get_history(limit)
    ]


@monitoring_router.post("/preemption/cycle")
def run_preemption_cycle() -> list[dict]:
    engine = _get_preemption_engine()
    actions = engine.cycle()
    return [
        {
            "workload_name": a.workload_name,
            "namespace": a.namespace,
            "priority": a.priority.value,
            "reason": a.reason,
            "status": a.status,
        }
        for a in actions
    ]


@monitoring_router.post("/preemption/start")
def start_preemption_engine() -> dict:
    _get_preemption_engine().start()
    return {"status": "started"}


@monitoring_router.post("/preemption/stop")
def stop_preemption_engine() -> dict:
    _get_preemption_engine().stop()
    return {"status": "stopped"}


@monitoring_router.get("/preemption/config")
def get_preemption_config() -> dict:
    engine = _get_preemption_engine()
    return {
        "policy": engine.config.policy.value,
        "min_priority_delta": engine.config.min_priority_delta,
        "preempt_oldest_first": engine.config.preempt_oldest_first,
        "max_preemptions_per_cycle": engine.config.max_preemptions_per_cycle,
    }


@monitoring_router.put("/preemption/config")
def update_preemption_config(data: dict[str, Any]) -> dict:
    engine = _get_preemption_engine()
    existing = engine.config
    policy_str = data.get("policy", existing.policy.value)
    engine.config = PreemptionPolicyConfig(
        policy=PreemptionPolicy(policy_str),
        min_priority_delta=data.get("min_priority_delta", existing.min_priority_delta),
        preempt_oldest_first=data.get("preempt_oldest_first", existing.preempt_oldest_first),
        max_preemptions_per_cycle=data.get("max_preemptions_per_cycle", existing.max_preemptions_per_cycle),
    )
    return {"status": "updated", "config": {
        "policy": engine.config.policy.value,
        "min_priority_delta": engine.config.min_priority_delta,
        "preempt_oldest_first": engine.config.preempt_oldest_first,
        "max_preemptions_per_cycle": engine.config.max_preemptions_per_cycle,
    }}


@monitoring_router.post("/preemption/apply")
def apply_preemption_policy(data: dict[str, Any]) -> dict:
    engine = _get_preemption_engine()
    success = engine.apply_preemption_policy(
        workload_name=data["workload_name"],
        namespace=data.get("namespace", "default"),
        priority=PriorityClass(data.get("priority", "batch")),
        policy=PreemptionPolicy(data.get("policy", "PreemptLowerPriority")),
    )
    return {"status": "applied" if success else "failed"}


@monitoring_router.get("/autoscaler/status")
def get_autoscaler_status() -> dict:
    status = _get_autoscaler_engine().get_status()
    return {
        "running": status.running,
        "policy": status.current_config.policy.value if status.current_config else "automatic",
        "last_event": {
            "direction": status.last_event.direction.value if status.last_event else None,
            "node_group": status.last_event.node_group if status.last_event else None,
            "reason": status.last_event.reason if status.last_event else None,
            "status": status.last_event.status if status.last_event else None,
        } if status.last_event else None,
        "event_count": status.event_count,
    }


@monitoring_router.get("/autoscaler/events")
def get_autoscaler_events(limit: int = 50) -> list[dict]:
    engine = _get_autoscaler_engine()
    return [
        {
            "direction": e.direction.value,
            "node_group": e.node_group,
            "delta": e.delta,
            "reason": e.reason,
            "initiated_at": e.initiated_at,
            "completed_at": e.completed_at,
            "status": e.status,
            "target_size": e.target_size,
        }
        for e in engine.get_events(limit)
    ]


@monitoring_router.post("/autoscaler/start")
def start_autoscaler() -> dict:
    _get_autoscaler_engine().start()
    return {"status": "started"}


@monitoring_router.post("/autoscaler/stop")
def stop_autoscaler() -> dict:
    _get_autoscaler_engine().stop()
    return {"status": "stopped"}


@monitoring_router.get("/autoscaler/config")
def get_autoscaler_config() -> dict:
    config = _get_autoscaler_engine().config
    return {
        "policy": config.policy.value,
        "scale_up_threshold": config.scale_up_threshold,
        "scale_down_threshold": config.scale_down_threshold,
        "scale_up_increment": config.scale_up_increment,
        "scale_down_decrement": config.scale_down_decrement,
        "cooldown_seconds": config.cooldown_seconds,
        "min_nodes": config.min_nodes,
        "max_nodes": config.max_nodes,
        "node_groups": [
            {
                "name": ng.name,
                "min_size": ng.min_size,
                "max_size": ng.max_size,
                "current_size": ng.current_size,
                "gpu_type": ng.gpu_type,
                "gpus_per_node": ng.gpus_per_node,
                "instance_type": ng.instance_type,
                "region": ng.region,
            }
            for ng in config.node_groups
        ],
    }


@monitoring_router.put("/autoscaler/config")
def update_autoscaler_config(data: dict[str, Any]) -> dict:
    engine = _get_autoscaler_engine()
    existing = engine.config
    node_groups_data = data.get("node_groups", [])
    node_groups = [
        NodeGroupConfig(
            name=ng.get("name", f"group-{i}"),
            min_size=ng.get("min_size", 1),
            max_size=ng.get("max_size", 10),
            current_size=ng.get("current_size", 1),
            gpu_type=ng.get("gpu_type", ""),
            gpus_per_node=ng.get("gpus_per_node", 8),
            instance_type=ng.get("instance_type", ""),
            region=ng.get("region", ""),
            labels=ng.get("labels", {}),
            taints=ng.get("taints", []),
        )
        for i, ng in enumerate(node_groups_data)
    ]
    engine.config = AutoscalingConfig(
        policy=ScalingPolicy(data.get("policy", existing.policy.value)),
        scale_up_threshold=data.get("scale_up_threshold", existing.scale_up_threshold),
        scale_down_threshold=data.get("scale_down_threshold", existing.scale_down_threshold),
        scale_up_increment=data.get("scale_up_increment", existing.scale_up_increment),
        scale_down_decrement=data.get("scale_down_decrement", existing.scale_down_decrement),
        cooldown_seconds=data.get("cooldown_seconds", existing.cooldown_seconds),
        min_nodes=data.get("min_nodes", existing.min_nodes),
        max_nodes=data.get("max_nodes", existing.max_nodes),
        node_groups=node_groups,
    )
    return {"status": "updated"}


@monitoring_router.post("/autoscaler/scale")
def manual_scale(data: dict[str, Any]) -> dict:
    engine = _get_autoscaler_engine()
    event = engine.scale_manual(
        node_group=data.get("node_group", "default-gpu-group"),
        target_size=data["target_size"],
    )
    return {
        "status": "scaled",
        "event": {
            "direction": event.direction.value,
            "node_group": event.node_group,
            "delta": event.delta,
            "reason": event.reason,
            "target_size": event.target_size,
            "status": event.status,
        },
    }
