"""Comprehensive data seeder -- populates ALL GPUOpt sectors with live mock data.

Usage:
    cd GPUOpt && PYTHONPATH=src python scripts/seed_comprehensive.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import random
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4

from gpuopt.config import get_settings
from gpuopt.repository import ClusterRepository
from gpuopt.schemas import (
    ActuationAction, ActuationRecord, ActuationStatus,
    AlertConditionType, AlertRecord, AlertRule, AlertSeverity,
    ApprovalRecord, ApprovalStatus, ApprovalStep,
    CheckItem, CheckStatus,
    ClusterCreate, ClusterStateData, ClusterTelemetry, ConnectorType,
    EnvironmentCheckReport,
    GPUDeviceState, GPUDeviceTelemetry,
    InferenceEndpoint, InferenceEndpointStatus, InferenceFramework,
    NodeState, NodeTelemetry,
    NotificationChannel, NotificationChannelType, NotificationMessage,
    PolicyRule, PolicySeverity, Project,
    RecommendationSet, RecommendationSeverity, RecommendationStatus, RecommendationType,
    ResourceRecommendation, ResourceQuota, Team,
    TrainingFramework, TrainingJob, TrainingJobStatus,
    TwinState, WorkloadAnalysisResult, GPUUtilizationTrend, NodeEfficiency,
    ChaosExperiment, ChaosFaultType, ChaosFaultTarget,
)

random.seed(42)
_NOW = datetime.now(timezone.utc)


def _ago(**kw):
    return _NOW - timedelta(**kw)


def _gpu_state(idx, model, total_gb, used_gb, status="healthy"):
    return GPUDeviceState(
        index=idx,
        uuid=f"GPU-{uuid4().hex[:12]}",
        model=model,
        memory_total_bytes=total_gb * 1024**3,
        memory_used_bytes=used_gb * 1024**3,
        status=status,
    )


def _gpu_telemetry(idx, util, mem_pct, temp, power, power_limit, model="NVIDIA A100 80GB"):
    return GPUDeviceTelemetry(
        index=idx,
        uuid=f"GPU-{uuid4().hex[:12]}",
        model=model,
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=int(80 * mem_pct / 100 * 1024**3),
        utilization_gpu_percent=util,
        utilization_memory_percent=mem_pct,
        temperature_gpu_celsius=temp,
        power_draw_watts=power,
        power_limit_watts=power_limit,
        ecc_errors_volatile=random.randint(0, 5),
        ecc_errors_aggregate=random.randint(0, 20),
        clock_sm_mhz=1410,
        clock_mem_mhz=1593,
    )


CLUSTER_CONFIGS = [
    {
        "name": "local-mock", "environment": "sandbox",
        "connector_type": ConnectorType.MOCK,
        "description": "Local mock GPU cluster for development & testing.",
        "region": "us-east-1",
        "nodes": [
            {"name": "mock-node-1", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("idle", 12), ("idle", 8), ("light", 4), ("idle", 2)],
             "telemetry": [(15, 18, 42, 185, 400), (8, 12, 38, 95, 400), (3, 5, 35, 55, 400), (1, 2, 33, 30, 400)]},
            {"name": "mock-node-2", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("medium", 40), ("medium", 35), ("light", 20), ("idle", 5)],
             "telemetry": [(55, 50, 62, 320, 400), (42, 44, 58, 280, 400), (25, 25, 48, 180, 400), (6, 6, 36, 70, 400)]},
        ],
    },
    {
        "name": "training-cluster", "environment": "staging",
        "connector_type": ConnectorType.MOCK,
        "description": "Staging cluster for ML training workloads (H100).",
        "region": "us-west-2",
        "nodes": [
            {"name": "train-node-1", "status": "Ready", "gpu_model": "NVIDIA H100 80GB",
             "gpus": [("heavy", 72), ("heavy", 65), ("heavy", 70), ("heavy", 68)],
             "telemetry": [(92, 90, 78, 520, 700), (88, 81, 75, 490, 700), (95, 88, 80, 550, 700), (90, 85, 77, 510, 700)]},
            {"name": "train-node-2", "status": "Ready", "gpu_model": "NVIDIA H100 80GB",
             "gpus": [("medium", 60), ("medium", 55), ("heavy", 58), ("medium", 62)],
             "telemetry": [(78, 75, 68, 420, 700), (72, 69, 65, 390, 700), (76, 72, 67, 405, 700), (81, 78, 70, 430, 700)]},
        ],
    },
    {
        "name": "production-inference", "environment": "production",
        "connector_type": ConnectorType.MOCK,
        "description": "Production inference serving cluster.",
        "region": "eu-west-1",
        "nodes": [
            {"name": "inf-node-1", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("heavy", 75), ("heavy", 72), ("heavy", 70), ("idle", 6), ("idle", 4)],
             "telemetry": [(97, 94, 82, 380, 400), (95, 90, 80, 370, 400), (93, 88, 78, 355, 400), (7, 8, 37, 60, 400), (4, 5, 35, 40, 400)]},
            {"name": "inf-node-2", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("heavy", 78), ("heavy", 76), ("heavy", 74), ("idle", 2)],
             "telemetry": [(98, 96, 85, 395, 400), (96, 95, 82, 385, 400), (94, 92, 79, 365, 400), (2, 3, 32, 25, 400)]},
            {"name": "inf-node-3", "status": "NotReady", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("failed", 80), ("failed", 80)],
             "telemetry": [(0, 100, 88, 450, 400), (0, 100, 86, 440, 400)]},
        ],
    },
    {
        "name": "rtx-4090-cluster", "environment": "production",
        "connector_type": ConnectorType.MOCK,
        "description": "High-performance RTX 4090 cluster for inference & rendering.",
        "region": "us-east-1",
        "nodes": [
            {"name": "rtx-node-1", "status": "Ready", "gpu_model": "NVIDIA RTX 4090",
             "gpus": [("medium", 12), ("heavy", 16), ("light", 4), ("medium", 10)],
             "telemetry": [(45, 38, 62, 280, 450), (78, 65, 72, 380, 450), (12, 8, 42, 120, 450), (52, 42, 58, 290, 450)]},
            {"name": "rtx-node-2", "status": "Ready", "gpu_model": "NVIDIA RTX 4090",
             "gpus": [("heavy", 20), ("medium", 14), ("heavy", 18), ("light", 6)],
             "telemetry": [(85, 78, 74, 410, 450), (62, 55, 65, 310, 450), (82, 75, 70, 400, 450), (8, 6, 40, 100, 450)]},
        ],
    },
    {
        "name": "ml-research-cluster", "environment": "sandbox",
        "connector_type": ConnectorType.MOCK,
        "description": "ML research cluster with varied GPU types.",
        "region": "us-west-2",
        "nodes": [
            {"name": "research-node-1", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("light", 8), ("medium", 30), ("idle", 2), ("medium", 25)],
             "telemetry": [(18, 10, 40, 120, 400), (38, 38, 55, 210, 400), (3, 3, 34, 35, 400), (35, 32, 52, 195, 400)]},
            {"name": "research-node-2", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("heavy", 48), ("idle", 3), ("light", 6), ("medium", 35)],
             "telemetry": [(72, 60, 68, 340, 400), (2, 4, 33, 30, 400), (15, 8, 42, 110, 400), (45, 44, 58, 250, 400)]},
        ],
    },
    {
        "name": "batch-processing", "environment": "staging",
        "connector_type": ConnectorType.MOCK,
        "description": "Staging cluster for batch ML training & data processing.",
        "region": "ap-southeast-1",
        "nodes": [
            {"name": "batch-node-1", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("heavy", 70), ("heavy", 65), ("medium", 45), ("medium", 40)],
             "telemetry": [(88, 85, 76, 480, 700), (82, 78, 72, 440, 700), (52, 50, 60, 290, 700), (48, 45, 58, 270, 700)]},
            {"name": "batch-node-2", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("heavy", 68), ("medium", 50)]},
            {"name": "batch-node-3", "status": "Ready", "gpu_model": "NVIDIA H100 80GB",
             "gpus": [("heavy", 72), ("heavy", 66), ("heavy", 70)]},
        ],
    },
    {
        "name": "dev-sandbox", "environment": "development",
        "connector_type": ConnectorType.MOCK,
        "description": "Developer sandbox for testing GPU workloads.",
        "region": "us-east-1",
        "nodes": [
            {"name": "dev-node-1", "status": "Ready", "gpu_model": "NVIDIA RTX 6000 Ada",
             "gpus": [("light", 4), ("light", 6), ("idle", 1), ("light", 3)],
             "telemetry": [(22, 12, 45, 150, 300), (15, 8, 40, 110, 300), (1, 2, 32, 25, 300), (18, 10, 42, 130, 300)]},
        ],
    },
    {
        "name": "hpc-cluster", "environment": "production",
        "connector_type": ConnectorType.MOCK,
        "description": "HPC cluster with H200 GPUs for large-scale training.",
        "region": "eu-central-1",
        "nodes": [
            {"name": "hpc-node-1", "status": "Ready", "gpu_model": "NVIDIA H200 141GB",
             "gpus": [("heavy", 120), ("heavy", 115), ("medium", 80), ("heavy", 130)],
             "telemetry": [(95, 94, 82, 620, 700), (92, 90, 78, 590, 700), (65, 60, 68, 410, 700), (96, 97, 84, 640, 700)]},
            {"name": "hpc-node-2", "status": "Ready", "gpu_model": "NVIDIA H200 141GB",
             "gpus": [("medium", 60), ("heavy", 110), ("heavy", 125), ("medium", 75)],
             "telemetry": [(55, 48, 60, 360, 700), (88, 85, 75, 560, 700), (94, 92, 80, 610, 700), (58, 52, 62, 380, 700)]},
        ],
    },
    {
        "name": "edge-inference", "environment": "production",
        "connector_type": ConnectorType.MOCK,
        "description": "Edge inference cluster with T4 GPUs for low-latency serving.",
        "region": "ap-northeast-1",
        "nodes": [
            {"name": "edge-node-1", "status": "Ready", "gpu_model": "NVIDIA T4",
             "gpus": [("medium", 8), ("medium", 10), ("light", 4), ("light", 6)],
             "telemetry": [(48, 38, 55, 52, 70), (52, 42, 58, 58, 70), (25, 18, 45, 35, 70), (30, 22, 48, 40, 70)]},
            {"name": "edge-node-2", "status": "Ready", "gpu_model": "NVIDIA T4",
             "gpus": [("heavy", 14), ("light", 5), ("medium", 10)]},
        ],
    },
    {
        "name": "finops-demo", "environment": "development",
        "connector_type": ConnectorType.MOCK,
        "description": "Demo cluster for FinOps cost analysis & optimization.",
        "region": "us-east-1",
        "nodes": [
            {"name": "finops-node-1", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("medium", 50), ("heavy", 72), ("medium", 45), ("light", 10)],
             "telemetry": [(55, 62, 62, 310, 400), (90, 88, 78, 385, 400), (48, 56, 60, 295, 400), (18, 14, 44, 140, 400)]},
            {"name": "finops-node-2", "status": "Ready", "gpu_model": "NVIDIA A100 80GB",
             "gpus": [("idle", 2), ("light", 8), ("heavy", 68), ("medium", 52)],
             "telemetry": [(2, 3, 32, 25, 400), (20, 12, 45, 150, 400), (85, 82, 74, 370, 400), (58, 55, 62, 305, 400)]},
        ],
    },
]


def _build_state(cluster_record, config):
    nodes, telemetry_nodes = [], []
    total_gpu_memory = 0

    for node_cfg in config["nodes"]:
        gpu_devices, telemetry_gpus = [], []
        for i, (usage_label, mem_used_gb) in enumerate(node_cfg["gpus"]):
            total_gb = 80
            model = node_cfg["gpu_model"]
            if "H200" in model:
                total_gb = 141
            is_failed = usage_label == "failed"
            gpu_devices.append(_gpu_state(i, model, total_gb, mem_used_gb, "failed" if is_failed else "healthy"))

            tel_data = node_cfg.get("telemetry")
            if tel_data and i < len(tel_data):
                util, mem_pct, temp, power, power_limit = tel_data[i]
            else:
                thresholds = {"idle": 2, "light": 10, "medium": 28, "heavy": 60, "failed": 80}
                util = thresholds.get(usage_label, 20) + random.uniform(-3, 3)
                mem_pct = (mem_used_gb / total_gb) * 100
                temp = 35 + util * 0.5 + random.uniform(-5, 5)
                pl = 400
                if "H100" in model or "H200" in model:
                    pl = 700
                elif "4090" in model:
                    pl = 450
                power = (util / 100) * 0.8 * pl + random.uniform(-20, 20)
                power_limit = pl

            telemetry_gpus.append(_gpu_telemetry(i, util, mem_pct, temp, power, power_limit, model))
            total_gpu_memory += total_gb * 1024**3

        nodes.append(NodeState(
            name=node_cfg["name"], status=node_cfg["status"],
            capacity={"cpu": "32", "memory": "256Gi", "pods": "110", "nvidia.com/gpu": str(len(gpu_devices))},
            allocatable={"cpu": "31", "memory": "250Gi", "pods": "110", "nvidia.com/gpu": str(len(gpu_devices))},
            labels={"kubernetes.io/hostname": node_cfg["name"], "node.kubernetes.io/instance-type": "gpu-standard", "nvidia.com/gpu.present": "true"},
            gpu_devices=gpu_devices, pod_count=len(gpu_devices), pod_capacity=110,
        ))
        telemetry_nodes.append(NodeTelemetry(
            node_name=node_cfg["name"], status=node_cfg["status"],
            cpu_usage_millicores=8000 + random.randint(0, 16000),
            cpu_capacity_millicores=32000,
            memory_usage_bytes=80 * 1024**3 + random.randint(0, 50) * 1024**3,
            memory_capacity_bytes=256 * 1024**3,
            pod_count=len(gpu_devices), pod_capacity=110,
            gpu_devices=telemetry_gpus,
        ))

    gpu_count = sum(len(n["gpus"]) for n in config["nodes"])
    telemetry = ClusterTelemetry(
        cluster_id=cluster_record.id, cluster_name=cluster_record.name,
        collected_at=_NOW, node_count=len(nodes), gpu_count=gpu_count,
        nodes=telemetry_nodes, freshness_seconds=0.0,
    )
    return ClusterStateData(
        cluster_id=cluster_record.id, cluster_name=cluster_record.name,
        environment=cluster_record.environment,
        collected_at=_NOW, generated_at=_NOW,
        node_count=len(nodes), gpu_count=gpu_count,
        total_gpu_memory_bytes=total_gpu_memory, nodes=nodes, telemetry=telemetry,
    )


def _make_trace(state, hours_ago, util_mult=1.0):
    import copy
    t = copy.deepcopy(state)
    t.collected_at = _ago(hours=hours_ago)
    t.generated_at = _ago(hours=hours_ago)
    if t.telemetry:
        t.telemetry.collected_at = t.collected_at
        for tn in t.telemetry.nodes:
            for tg in tn.gpu_devices:
                tg.utilization_gpu_percent = min(100, max(0, tg.utilization_gpu_percent * util_mult + random.uniform(-5, 5)))
                tg.temperature_gpu_celsius = min(95, max(30, tg.temperature_gpu_celsius * (0.9 + 0.2 * util_mult)))
    return t


def _make_recommendations(cluster_id, cluster_name, count=5):
    templates = [
        (RecommendationType.PLACEMENT, RecommendationSeverity.HIGH,
         "Consolidate idle GPUs", "Consolidate workloads to free underutilized GPUs",
         "Reduce node count, maintain throughput", "~$4,200/mo savings, 25% util improvement", 0.82),
        (RecommendationType.RIGHT_SIZING, RecommendationSeverity.MEDIUM,
         "Right-size GPU allocations", "Several jobs over-provisioned on GPU memory",
         "Reduce GPU memory requests by 30% on light workloads", "$1,800/mo waste reduction", 0.71),
        (RecommendationType.SCALING, RecommendationSeverity.LOW,
         "Enable auto-scaling for inference", "Inference endpoints benefit from HPA-based scaling",
         "Scale down during low-traffic periods (22:00-06:00)", "~$2,500/mo savings", 0.65),
        (RecommendationType.RISK_MITIGATION, RecommendationSeverity.CRITICAL,
         "Migrate from degrading GPU", "GPU has ECC errors and temperature > 85C",
         "Immediate workload migration", "Prevent job failures and data corruption", 0.94),
        (RecommendationType.EFFICIENCY, RecommendationSeverity.HIGH,
         "Enable GPU sharing with MIG", "MIG-capable GPUs can be partitioned",
         "Configure MIG profiles for sharing", "3x GPU utilization for small workloads", 0.78),
    ]
    recs = []
    for i, (rt, rs, title, desc, reason, impact, conf) in enumerate(templates[:count]):
        recs.append(ResourceRecommendation(
            type=rt, severity=rs, title=title, description=desc,
            reasoning=reason, expected_impact=impact, confidence=conf,
            risk_level="low" if rs in (RecommendationSeverity.LOW, RecommendationSeverity.MEDIUM) else "medium",
            affected_resources=[f"{cluster_name}/node-{i % 3 + 1}"],
            actions=[f"Apply {rt.value}: {title.lower()}"],
            estimated_savings={"gpu_hours_monthly": 120 + i * 30, "cost_usd_monthly": 1200 + i * 600},
            score=round(95 - i * 18 + random.uniform(-5, 5), 1),
            status=RecommendationStatus.PENDING if i > 0 else RecommendationStatus.APPROVED,
        ))
    return RecommendationSet(
        cluster_id=cluster_id, cluster_name=cluster_name, environment="production",
        based_on_state_at=_ago(hours=1),
        recommendation_count=len(recs),
        critical_count=sum(1 for r in recs if r.severity == RecommendationSeverity.CRITICAL),
        high_count=sum(1 for r in recs if r.severity == RecommendationSeverity.HIGH),
        recommendations=recs,
        summary=f"{len(recs)} recommendations for {cluster_name}",
        avg_score=round(sum(r.score for r in recs) / len(recs), 1),
        total_estimated_savings_gpu_hours=sum(r.estimated_savings.get("gpu_hours_monthly", 0) for r in recs),
        top_recommendation=recs[0].title,
    )


def _make_analysis(cluster_id, cluster_name, gpu_count, node_count):
    trends = []
    for i in range(min(gpu_count, 10)):
        trends.append(GPUUtilizationTrend(
            gpu_uuid=f"GPU-{uuid4().hex[:12]}", node=f"node-{i % node_count + 1}",
            gpu_index=i, model="NVIDIA A100 80GB", memory_total_bytes=80 * 1024**3,
            avg_utilization_percent=round(40 + random.random() * 40, 1),
            peak_utilization_percent=round(70 + random.random() * 28, 1),
            min_utilization_percent=round(random.random() * 15, 1),
            memory_pressure_percent=round(random.random() * 60, 1),
            idle_percent=round(random.random() * 40, 1),
            sample_count=random.randint(10, 100),
        ))
    nodes = []
    for i in range(node_count):
        gu = 30 + random.random() * 50
        nodes.append(NodeEfficiency(
            node_name=f"node-{i+1}", status="Ready",
            gpu_count=gpu_count // node_count,
            avg_gpu_utilization_percent=round(gu, 1),
            gpu_idle_percent=round(max(0, 100 - gu - 20), 1),
            avg_memory_utilization_percent=round(gu * 0.8, 1),
            efficiency_score=round(gu / 100, 2),
            recommendations=["Right-size GPU allocation"] if gu < 40 else [],
        ))
    oe = round(sum(n.efficiency_score for n in nodes) / len(nodes), 2) if nodes else 0.5
    return WorkloadAnalysisResult(
        cluster_id=cluster_id, cluster_name=cluster_name,
        generated_at=_NOW, timeframe_hours=168,
        trace_count=random.randint(10, 50),
        node_count=node_count, gpu_count=gpu_count,
        total_gpu_hours=round(gpu_count * 168 * random.uniform(0.5, 0.9), 0),
        gpu_trends=trends, node_efficiencies=nodes,
        overall_efficiency_score=oe,
        total_idle_gpu_hours=round(gpu_count * 168 * (1 - oe), 0),
        estimated_power_waste_kwh=round(gpu_count * 168 * 0.4 * (1 - oe), 0),
        summary=f"Overall efficiency: {oe:.0%} across {gpu_count} GPUs",
    )


def _make_check_report(cluster_id, cluster_name, env, status_override=None):
    st = status_override or random.choice([CheckStatus.PASS, CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL])
    checks = [
        CheckItem(name="api_server", status=CheckStatus.PASS, message="API server is available."),
        CheckItem(name="gpu_operator", status=CheckStatus.PASS, message=f"GPU operator managing {random.randint(4, 16)} GPUs."),
        CheckItem(name="dcgm_exporter", status=CheckStatus.PASS, message="DCGM exporter reporting."),
        CheckItem(name="node_inventory", status=st, message="All nodes Ready." if st == CheckStatus.PASS else "Some nodes NotReady."),
        CheckItem(name="gpu_inventory", status=CheckStatus.PASS, message=f"{random.randint(4, 27)} GPUs discovered."),
        CheckItem(name="prometheus", status=CheckStatus.PASS, message="Prometheus scraping."),
        CheckItem(name="rbac_permissions", status=CheckStatus.PASS, message="RBAC sufficient."),
    ]
    summary = {s.value: sum(1 for c in checks if c.status == s) for s in CheckStatus}
    return EnvironmentCheckReport(
        cluster_id=cluster_id, cluster_name=cluster_name, environment=env,
        started_at=_ago(hours=random.randint(1, 48)),
        completed_at=_ago(hours=random.randint(0, 47)),
        overall_status=st, checks=checks, summary=summary,
    )


def _make_actuation(cluster_id, cluster_name, rec_id, status_override=None):
    st = status_override or random.choice([ActuationStatus.COMPLETED, ActuationStatus.COMPLETED, ActuationStatus.FAILED, ActuationStatus.IN_PROGRESS, ActuationStatus.ROLLED_BACK])
    ca = _ago(hours=random.randint(1, 48)) if st in (ActuationStatus.COMPLETED, ActuationStatus.FAILED, ActuationStatus.ROLLED_BACK) else None
    actions = [
        ActuationAction(action_type="evacuate_pods", target="node-1", value="3 pods migrated", status="completed", message="Migrated 3 inference pods"),
        ActuationAction(action_type="power_off_gpu", target="gpu-2", value="GPU powered off", status="completed" if st == ActuationStatus.COMPLETED else "failed", message="GPU power off " + ("ok" if st == ActuationStatus.COMPLETED else "failed")),
    ]
    return ActuationRecord(
        cluster_id=cluster_id, cluster_name=cluster_name, rec_id=rec_id,
        rec_title="Consolidate idle GPUs", rec_type="placement",
        status=st, dry_run=False,
        started_at=_ago(hours=random.randint(2, 72)), completed_at=ca,
        actions=actions,
        result_summary="Successfully consolidated" if st == ActuationStatus.COMPLETED else "Failed to consolidate" if st == ActuationStatus.FAILED else "Rolled back consolidation",
        error_message="" if st in (ActuationStatus.COMPLETED, ActuationStatus.IN_PROGRESS) else "GPU 2 failed to power off",
    )


def _make_approval(actuation_id, cluster_id, cluster_name, status_override=None):
    st = status_override or random.choice([ApprovalStatus.APPROVED, ApprovalStatus.APPROVED, ApprovalStatus.PENDING, ApprovalStatus.REJECTED])
    steps = [
        ApprovalStep(step_order=1, approver="admin@example.com",
                     status=ApprovalStatus.APPROVED if st != ApprovalStatus.REJECTED else ApprovalStatus.REJECTED,
                     decided_at=_ago(hours=random.randint(1, 24)),
                     reason="Approved - low risk" if st != ApprovalStatus.REJECTED else "Rejected - insufficient testing"),
    ]
    return ApprovalRecord(
        actuation_id=actuation_id, cluster_id=cluster_id, cluster_name=cluster_name,
        status=st, steps=steps, required_approvers=["admin@example.com"],
        reason="Schedule maintenance window",
        created_at=_ago(hours=random.randint(24, 72)),
        decided_at=_ago(hours=random.randint(1, 23)) if st != ApprovalStatus.PENDING else None,
        final_reason="" if st == ApprovalStatus.PENDING else steps[0].reason,
    )


def _make_alert_record(rule_id, cluster_id, cluster_name, severity, condition, value, threshold):
    return AlertRecord(
        rule_id=rule_id, cluster_id=cluster_id, cluster_name=cluster_name,
        severity=severity, condition_type=condition,
        current_value=value, threshold=threshold,
        message=f"{condition.value} is {value:.1f} (threshold: {threshold})",
        status="firing",
    )


def main():
    settings = get_settings()
    settings.database_path = Path("data/gpuopt.db")
    repo = ClusterRepository(settings.database_path)

    print(f"Database: {settings.database_path}")
    print("=" * 60)

    all_clusters = []

    # Step 1: Create clusters
    print("\n[Step 1/17] Creating 10 clusters...")
    for cfg in CLUSTER_CONFIGS:
        payload = ClusterCreate(
            name=cfg["name"], environment=cfg["environment"],
            connector_type=cfg["connector_type"], description=cfg["description"],
            region=cfg.get("region"),
        )
        record = repo.upsert_cluster(payload)
        all_clusters.append((record, cfg))
        total_gpus = sum(len(n["gpus"]) for n in cfg["nodes"])
        total_nodes = len(cfg["nodes"])
        print(f"  [OK] {record.environment}/{record.name} - {total_nodes} nodes, {total_gpus} GPUs")

    # Step 2: State snapshots
    print("\n[Step 2/17] Seeding state snapshots (traces)...")
    for record, cfg in all_clusters:
        state = _build_state(record, cfg)
        repo.save_state(state)
        for ha, um in [(6, 0.7), (12, 1.1), (24, 0.5), (48, 1.3), (72, 0.9), (168, 1.0)]:
            repo.save_state(_make_trace(state, ha, um))
        print(f"  [OK] {record.name}: 7 traces")

    # Step 3: Baselines
    print("\n[Step 3/17] Setting baselines...")
    for record, cfg in all_clusters:
        traces = repo.list_traces(record.id, limit=5)
        if traces:
            repo.set_baseline(record.id, traces[0][1], traces[0][0])
        print(f"  [OK] Baseline for {record.name}")

    # Step 4: Analyses
    print("\n[Step 4/17] Seeding workload analyses...")
    for record, cfg in all_clusters:
        gpu_c = sum(len(n["gpus"]) for n in cfg["nodes"])
        node_c = len(cfg["nodes"])
        analysis = _make_analysis(record.id, record.name, gpu_c, node_c)
        repo.save_analysis(analysis)
        print(f"  [OK] {record.name}: {analysis.gpu_count} GPUs, eff={analysis.overall_efficiency_score:.0%}")

    # Step 5: Recommendations
    print("\n[Step 5/17] Seeding recommendations...")
    for record, cfg in all_clusters:
        rec_set = _make_recommendations(record.id, record.name, count=random.randint(3, 6))
        repo.save_recommendations(rec_set)
        print(f"  [OK] {record.name}: {rec_set.recommendation_count} recs, score={rec_set.avg_score:.0f}")

    # Step 6: Digital twins
    print("\n[Step 6/17] Seeding digital twin records...")
    for record, cfg in all_clusters:
        state = repo.latest_state(record.id)
        if state:
            twin = TwinState(
                cluster_id=record.id, cluster_name=record.name,
                environment=record.environment, synced_at=_NOW,
                original_collected_at=state.collected_at,
                node_count=state.node_count, gpu_count=state.gpu_count,
                state_json=state.model_dump_json(),
                has_diverged=random.random() < 0.3,
                divergence_reason="Telemetry drift detected" if random.random() < 0.3 else "",
            )
            repo.save_twin(twin)
            print(f"  [{'DIVERGED' if twin.has_diverged else 'SYNCED'}] {record.name}")

    # Step 7: Actuations
    print("\n[Step 7/17] Seeding actuation records...")
    for record, cfg in all_clusters:
        rec = repo.latest_recommendations(record.id)
        rec_id = rec.recommendations[0].id if rec and rec.recommendations else uuid4()
        for i in range(random.randint(2, 5)):
            repo.save_actuation(_make_actuation(record.id, record.name, rec_id))
        print(f"  [OK] {record.name}")

    # Step 8: Check reports
    print("\n[Step 8/17] Seeding environment check reports...")
    for record, cfg in all_clusters:
        for i in range(random.randint(3, 6)):
            st = CheckStatus.PASS if random.random() < 0.6 else (CheckStatus.WARN if random.random() < 0.5 else CheckStatus.FAIL)
            repo.save_report(_make_check_report(record.id, record.name, record.environment, st))
        print(f"  [OK] {record.name}")

    # Step 9: Policies
    print("\n[Step 9/17] Seeding guardrail policies...")
    policy_defs = [
        ("production-only-deploy", "Only allow deployments to production", "environment_restriction", PolicySeverity.CRITICAL, "block"),
        ("maintenance-window", "Block deployments 02:00-04:00", "time_window", PolicySeverity.HIGH, "block"),
        ("gpu-limit-32", "Max 32 GPUs per tenant", "resource_limit", PolicySeverity.MEDIUM, "warn"),
        ("approval-critical-changes", "Require approval for critical changes", "approval_required", PolicySeverity.CRITICAL, "block"),
        ("backup-before-actuation", "Require backup before actuation", "custom", PolicySeverity.MEDIUM, "warn"),
        ("cross-region-migration", "Only migrate within same region", "environment_restriction", PolicySeverity.HIGH, "block"),
        ("auto-reclaim-idle", "Auto-reclaim GPUs idle >2h", "custom", PolicySeverity.LOW, "allow"),
    ]
    for name, desc, rt, sev, fa in policy_defs:
        repo.save_policy(PolicyRule(name=name, description=desc, scope_type="global", rule_config={}, rule_type=rt, severity=sev, enabled=True, fail_action=fa))
    print(f"  [OK] {len(policy_defs)} policies")

    # Step 10: Approvals
    print("\n[Step 10/17] Seeding approval records...")
    for record, cfg in all_clusters[:5]:
        acts = repo.list_actuations(record.id, limit=1)
        if acts:
            repo.save_approval(_make_approval(acts[0].id, record.id, record.name))
        print(f"  [OK] {record.name}")

    # Step 11: Alerts
    print("\n[Step 11/17] Seeding alert rules and records...")
    for record, cfg in all_clusters:
        rule_specs = [
            ("GPU util critical", AlertConditionType.GPU_UTILIZATION, "lt", 10, AlertSeverity.CRITICAL),
            ("Temperature warning", AlertConditionType.GPU_TEMPERATURE, "gt", 80, AlertSeverity.WARNING),
            ("Idle GPU alert", AlertConditionType.IDLE_GPU, "gt", 2, AlertSeverity.WARNING),
            ("Memory pressure", AlertConditionType.MEMORY_UTILIZATION, "gt", 90, AlertSeverity.CRITICAL),
            ("Power efficiency", AlertConditionType.POWER_EFFICIENCY, "lt", 0.3, AlertSeverity.WARNING),
        ]
        for name, cond, op, thresh, sev in rule_specs:
            rule = AlertRule(name=name, description=f"Alert when {cond.value} {op} {thresh}", cluster_id=record.id, condition_type=cond, operator=op, threshold=thresh, severity=sev, enabled=True)
            repo.save_alert_rule(rule)
            alert = _make_alert_record(rule.id, record.id, record.name, sev, cond, random.uniform(2, 15) if cond == AlertConditionType.GPU_UTILIZATION else random.uniform(75, 92), thresh)
            repo.save_alert_record(alert)
        print(f"  [OK] {record.name}: 5 rules, 5 alerts")

    # Step 12: Notification channels
    print("\n[Step 12/17] Seeding notification channels...")
    channels = [
        NotificationChannel(name="Slack #gpu-alerts", channel_type=NotificationChannelType.SLACK, config={"webhook_url": "https://hooks.slack.com/services/T00/B00/xxx"}, enabled=True),
        NotificationChannel(name="Email Ops Team", channel_type=NotificationChannelType.EMAIL, config={"to": ["ops@example.com"], "from": "gpuopt@example.com"}, enabled=True),
        NotificationChannel(name="PagerDuty Critical", channel_type=NotificationChannelType.PAGERDUTY, config={"routing_key": "pagerduty_key"}, enabled=True),
    ]
    for ch in channels:
        repo.save_notification_channel(ch)
        for i in range(3):
            repo.save_notification_message(NotificationMessage(
                channel_id=ch.id, channel_name=ch.name,
                subject=f"GPU Alert #{i+1}: {'Low utilization' if i==0 else 'High temperature' if i==1 else 'Weekly report'}",
                body="Cluster production-inference has GPUs below 10% utilization.",
                status=random.choice(["sent", "sent", "sent", "failed"]),
                sent_at=_ago(hours=random.randint(1, 48)),
            ))
    print(f"  [OK] {len(channels)} channels, {len(channels)*3} messages")

    # Step 13: Teams & projects
    print("\n[Step 13/17] Seeding multi-tenancy data...")
    team_names = [("ml-engineering", "ML Engineering"), ("data-science", "Data Science"), ("platform-ops", "Platform Ops")]
    team_ids = []
    for slug, name in team_names:
        repo.save_policy(PolicyRule(name=f"team-{slug}", description=f"Team: {name}", scope_type="cluster", rule_type="custom", severity=PolicySeverity.LOW, enabled=True, fail_action="allow", rule_config={"team_name": name}))
        team_ids.append(uuid4())
        print(f"  [OK] Team: {name}")

    projects = [
        ("LLM Training", team_ids[0], [all_clusters[1][0].id, all_clusters[6][0].id]),
        ("Model Serving", team_ids[0], [all_clusters[2][0].id, all_clusters[3][0].id]),
        ("Research", team_ids[1], [all_clusters[4][0].id, all_clusters[6][0].id]),
        ("Platform Monitoring", team_ids[2], [c[0].id for c in all_clusters]),
        ("FinOps Analysis", team_ids[2], [all_clusters[9][0].id]),
    ]
    for pname, tid, cids in projects:
        repo.save_policy(PolicyRule(name=f"project-{pname.lower().replace(' ', '-')}", description=f"Project: {pname}", scope_type="cluster", rule_type="custom", severity=PolicySeverity.LOW, enabled=True, fail_action="allow", rule_config={"project_name": pname, "team_id": str(tid), "cluster_ids": [str(c) for c in cids], "quota": {"max_gpus": 32, "max_monthly_cost": 50000}}))
    print(f"  [OK] {len(projects)} projects")

    # Step 14: Chaos experiments
    print("\n[Step 14/17] Seeding chaos experiments...")
    for record, cfg in all_clusters[:4]:
        exp = ChaosExperiment(cluster_id=record.id, cluster_name=record.name, name=f"GPU failure test - {record.name}", description="Simulate GPU failure", fault_type=ChaosFaultType.GPU_FAILURE, target=ChaosFaultTarget(target_type="gpu", target_selector={"gpu_index": 0}), duration_seconds=120, intensity=0.7, status="completed", started_at=_ago(hours=random.randint(12, 72)), completed_at=_ago(hours=random.randint(1, 11)))
        repo.save_actuation(ActuationRecord(cluster_id=record.id, cluster_name=record.name, rec_id=uuid4(), rec_title=f"Chaos: {exp.name}", rec_type="chaos_experiment", status=ActuationStatus.COMPLETED, dry_run=False, started_at=exp.started_at or _NOW, completed_at=exp.completed_at, actions=[ActuationAction(action_type="chaos_inject", target=f"gpu-0", value=exp.fault_type.value, status="completed", message=f"Injected {exp.fault_type.value} for {exp.duration_seconds}s")], result_summary="Chaos experiment completed: system resilient"))
        print(f"  [OK] {record.name}")

    # Step 15: Training jobs
    print("\n[Step 15/17] Seeding training jobs...")
    jobs = [
        (all_clusters[1][0], "llm-fine-tune-1", TrainingFramework.PYTORCH, TrainingJobStatus.RUNNING, 4, "NVIDIA H100 80GB"),
        (all_clusters[1][0], "gpt-training-v2", TrainingFramework.PYTORCH, TrainingJobStatus.COMPLETED, 8, "NVIDIA H100 80GB"),
        (all_clusters[5][0], "batch-inference-job", TrainingFramework.TENSORFLOW, TrainingJobStatus.RUNNING, 2, "NVIDIA A100 80GB"),
        (all_clusters[5][0], "data-pipeline", TrainingFramework.CUSTOM, TrainingJobStatus.PENDING, 4, "NVIDIA A100 80GB"),
        (all_clusters[4][0], "hpo-experiment", TrainingFramework.JAX, TrainingJobStatus.RUNNING, 2, "NVIDIA A100 80GB"),
        (all_clusters[7][0], "large-scale-train-1", TrainingFramework.PYTORCH, TrainingJobStatus.RUNNING, 8, "NVIDIA H200 141GB"),
        (all_clusters[7][0], "dist-train-test", TrainingFramework.PYTORCH, TrainingJobStatus.COMPLETED, 16, "NVIDIA H200 141GB"),
        (all_clusters[6][0], "dev-test-run", TrainingFramework.PYTORCH, TrainingJobStatus.COMPLETED, 1, "NVIDIA RTX 6000 Ada"),
        (all_clusters[3][0], "rtx-render-job", TrainingFramework.CUSTOM, TrainingJobStatus.RUNNING, 4, "NVIDIA RTX 4090"),
        (all_clusters[2][0], "llama-serving-test", TrainingFramework.CUSTOM, TrainingJobStatus.COMPLETED, 2, "NVIDIA A100 80GB"),
    ]
    for record, jname, fw, st, gc, gm in jobs:
        ej = random.randint(1, 48) if st == TrainingJobStatus.RUNNING else (random.randint(48, 168) if st == TrainingJobStatus.COMPLETED else 0)
        repo.save_actuation(ActuationRecord(cluster_id=record.id, cluster_name=record.name, rec_id=uuid4(), rec_title=f"Training: {jname}", rec_type="training_job", status=ActuationStatus.COMPLETED if st == TrainingJobStatus.COMPLETED else (ActuationStatus.IN_PROGRESS if st == TrainingJobStatus.RUNNING else ActuationStatus.PENDING), dry_run=False, started_at=_ago(hours=ej or 1), actions=[ActuationAction(action_type="start_training", target=jname, value=f"{gc}x {gm}", status="completed", message=f"Training started: {gc} GPUs")], result_summary=f"Training {jname}: {gc}x {gm}, {st.value}"))
    print(f"  [OK] {len(jobs)} training jobs")

    # Step 16: Inference endpoints
    print("\n[Step 16/17] Seeding inference endpoints...")
    endpoints = [
        (all_clusters[2][0], "llama-3-70b", "meta-llama/Llama-3-70b-chat", InferenceFramework.VLLM, InferenceEndpointStatus.RUNNING, 4, "NVIDIA A100 80GB"),
        (all_clusters[2][0], "mistral-serve", "mistralai/Mistral-7B", InferenceFramework.TGI, InferenceEndpointStatus.RUNNING, 2, "NVIDIA A100 80GB"),
        (all_clusters[2][0], "gpt-j-6b", "EleutherAI/gpt-j-6B", InferenceFramework.TENSORRTLLM, InferenceEndpointStatus.SCALING, 1, "NVIDIA A100 80GB"),
        (all_clusters[9][0], "stable-diffusion", "stabilityai/stable-diffusion-xl", InferenceFramework.TENSORRTLLM, InferenceEndpointStatus.RUNNING, 2, "NVIDIA A100 80GB"),
        (all_clusters[3][0], "whisper-large", "openai/whisper-large-v3", InferenceFramework.LLAMACPP, InferenceEndpointStatus.RUNNING, 1, "NVIDIA RTX 4090"),
        (all_clusters[8][0], "tiny-llama-edge", "TinyLlama/TinyLlama-1.1B", InferenceFramework.LLAMACPP, InferenceEndpointStatus.RUNNING, 1, "NVIDIA T4"),
        (all_clusters[8][0], "whisper-edge", "openai/whisper-small", InferenceFramework.LLAMACPP, InferenceEndpointStatus.DEPLOYING, 1, "NVIDIA T4"),
        (all_clusters[7][0], "h200-llm-serve", "meta-llama/Llama-3-70b-chat", InferenceFramework.VLLM, InferenceEndpointStatus.RUNNING, 8, "NVIDIA H200 141GB"),
        (all_clusters[5][0], "batch-llm", "mistralai/Mixtral-8x7B", InferenceFramework.VLLM, InferenceEndpointStatus.RUNNING, 4, "NVIDIA A100 80GB"),
        (all_clusters[0][0], "mock-endpoint", "gpt2", InferenceFramework.CUSTOM, InferenceEndpointStatus.RUNNING, 1, "NVIDIA A100 80GB"),
    ]
    for record, name, model, fw, st, gc, gm in endpoints:
        lat = round(random.uniform(50, 500), 1)
        tput = round(random.uniform(5, 100), 1)
        repo.save_actuation(ActuationRecord(cluster_id=record.id, cluster_name=record.name, rec_id=uuid4(), rec_title=f"Inference: {name}", rec_type="inference_endpoint", status=ActuationStatus.COMPLETED if st in (InferenceEndpointStatus.RUNNING, InferenceEndpointStatus.SCALING) else ActuationStatus.PENDING, dry_run=False, started_at=_ago(hours=random.randint(12, 168)), actions=[ActuationAction(action_type="deploy_endpoint", target=name, value=f"{gc}x {gm}", status="completed", message=f"Deployed {model} on {gc} GPUs")], result_summary=f"Endpoint {name}: {model}, {gc}x {gm}, {lat}ms latency, {tput} rps"))
    print(f"  [OK] {len(endpoints)} inference endpoints")

    # Step 17: Additional time-series snapshots
    print("\n[Step 17/17] Seeding additional data snapshots...")
    for record, cfg in all_clusters:
        for days_ago, lp in [(2, 0.3), (4, 1.2), (7, 0.6), (14, 0.8), (30, 0.4)]:
            repo.save_state(_make_trace(_build_state(record, cfg), hours_ago=days_ago * 24, util_mult=lp))
        print(f"  [OK] {record.name}: +5 snapshots")

    # Summary
    print("\n" + "=" * 60)
    print("SEEDING SUMMARY")
    print("=" * 60)
    total_gpus = sum(sum(len(n["gpus"]) for n in cfg["nodes"]) for _, cfg in all_clusters)
    total_nodes = sum(len(cfg["nodes"]) for _, cfg in all_clusters)
    print(f"  Total: {len(all_clusters)} clusters, {total_nodes} nodes, {total_gpus} GPUs")

    with repo._backend.connect() as conn:
        cur = conn.execute("SELECT COUNT(*) as cnt FROM cluster_state"); traces = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM analyses"); analyses = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM recommendations"); recs = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM actuations"); acts = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM check_reports"); checks = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM policies"); policies = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM approvals"); approvals = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM alert_rules"); arules = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM alert_records"); arecs = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM notification_channels"); chans = cur.fetchone()["cnt"]
        cur = conn.execute("SELECT COUNT(*) as cnt FROM notification_messages"); msgs = cur.fetchone()["cnt"]

    print(f"\n  Data seeded:")
    print(f"    Traces:         {traces}")
    print(f"    Analyses:       {analyses}")
    print(f"    Recommendations:{recs}")
    print(f"    Actuations:     {acts}")
    print(f"    Checks:         {checks}")
    print(f"    Policies:       {policies}")
    print(f"    Approvals:      {approvals}")
    print(f"    Alert Rules:    {arules}")
    print(f"    Alert Records:  {arecs}")
    print(f"    Notif Channels: {chans}")
    print(f"    Notif Messages: {msgs}")
    print(f"    Training Jobs:  {len(jobs)}")
    print(f"    Inference:      {len(endpoints)}")
    print()
    print("SEEDING COMPLETE!")
    print("  Server: cd GPUOpt && PYTHONPATH=src uvicorn gpuopt.main:app --host 0.0.0.0 --port 8082")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
