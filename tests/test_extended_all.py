from __future__ import annotations

from src.gpuopt.digital_twin_extended import (
    CandidateActionScorer, CostSloPowerSimulator, CounterfactualEngine, ExtendedTwinService,
)
from src.gpuopt.governance_extended import (
    ExplanationGenerator, ExtendedGovernanceService, PolicyEnforcer,
    ReportGenerator, RollbackManager, TenantQuotaManager,
)
from src.gpuopt.inference_extended import (
    ExtendedInferenceService, MoEOptimizer, ModelInstancePlacer,
    ReplicaRightSizer, RoutingRecommender, SloAwareScaler,
)
from src.gpuopt.integration_extended import (
    AiRuntimeDetector, IntegrationManager, ObjectStoreConnector,
    OpenTelemetryIntegrator, PrometheusIntegrator,
)
from src.gpuopt.optimization_extended import (
    ConsolidationPlanner, ElasticWorkerOptimizer,
    ExtendedOptimizationService, GpuTierSelector, RecommendationPrioritizer,
)
from src.gpuopt.prediction_extended import (
    ActionImpactForecaster, ComprehensivePredictionService,
    DemandBurstDetector, JCTPredictor, OOMRiskPredictor,
    QueuePressurePredictor, ThermalRiskPredictor,
)
from src.gpuopt.telemetry_extended import (
    ExtendedTelemetryService, FabricTelemetryCollector,
    JobTelemetryCollector, ModelServiceCollector,
    QueueTelemetryCollector, TelemetryStreamer,
)
from src.gpuopt.training_extended import (
    CheckpointManager, ElasticScalingPlanner, ExtendedTrainingService,
    HPOManager, HeterogeneousGpuAssigner, QueueAwarePlacementEngine,
)


# ── Telemetry & State Tests ────────────────────────────────────

def test_model_service_collector():
    c = ModelServiceCollector()
    tele = c.collect("test-model", num_endpoints=2)
    assert tele.model_name == "test-model"
    assert len(tele.endpoints) == 2
    assert tele.endpoints[0].total_requests > 0


def test_fabric_telemetry_collector():
    c = FabricTelemetryCollector()
    fab = c.collect("node-0", 0, 4)
    assert fab.node == "node-0"
    assert len(fab.links) == 4
    assert fab.nvlink_bandwidth_utilization >= 0


def test_queue_telemetry_collector():
    c = QueueTelemetryCollector()
    queues = c.collect("test", 3)
    assert len(queues) == 3
    assert queues[0].queue_name == "test-0"
    assert queues[0].queue_depth >= 0


def test_job_telemetry_collector():
    c = JobTelemetryCollector()
    jobs = c.collect(5)
    assert len(jobs) == 5
    assert jobs[0].job_id != ""


def test_extended_telemetry_service():
    s = ExtendedTelemetryService()
    snap = s.collect_full_snapshot("test-cluster")
    assert snap.cluster_id == "test-cluster"
    assert snap.snapshot_id != ""
    assert len(snap.queues) > 0


def test_telemetry_streamer():
    s = TelemetryStreamer()
    assert len(s.get_events()) == 0
    s.start(0.1)
    import time
    time.sleep(0.3)
    events = s.get_events()
    assert len(events) > 0
    s.stop()


# ── Prediction Tests ──────────────────────────────────────────

def test_queue_pressure_predictor():
    p = QueuePressurePredictor()
    result = p.predict()
    assert result.current_queue_depth >= 0
    assert result.predicted_queue_depth >= 0
    assert result.pressure_level in ("low", "medium", "high", "critical")


def test_jct_predictor():
    p = JCTPredictor()
    result = p.predict({"job_id": "test-job", "gpu_required": 2, "memory_required_gb": 16})
    assert result.estimated_duration_minutes > 0
    assert result.p50_duration_minutes > 0
    assert result.confidence > 0


def test_oom_risk_predictor():
    p = OOMRiskPredictor()
    result = p.predict({"job_id": "test", "memory_required_gb": 64}, {"devices": [{"index": 0, "memory_total_mb": 81920, "memory_used_mb": 40960}]})
    assert result.oom_probability >= 0
    assert result.risk_level in ("low", "medium", "high")


def test_thermal_risk_predictor():
    p = ThermalRiskPredictor()
    results = p.predict({"devices": [{"index": 0, "temperature_celsius": 75, "utilization_percent": 80, "power_draw_watts": 300}]})
    assert len(results) > 0
    assert results[0].current_temperature_c > 0


def test_demand_burst_detector():
    d = DemandBurstDetector()
    result = d.detect([10, 12, 11, 13, 10, 11, 12, 90, 95, 85])
    assert result.burst_detected or not result.burst_detected
    assert result.burst_magnitude > 0


def test_action_impact_forecaster():
    f = ActionImpactForecaster()
    result = f.forecast("consolidate")
    assert result.action_type == "consolidate"
    assert result.expected_cost_savings != 0


def test_comprehensive_prediction_service():
    s = ComprehensivePredictionService()
    result = s.predict_all()
    assert result.prediction_id != ""
    assert result.overall_risk_score >= 0


# ── Digital Twin Tests ────────────────────────────────────────

def test_counterfactual_engine():
    e = CounterfactualEngine()
    s = e.create_scenario("test", "Test scenario", [{"action_type": "scale_down"}])
    assert s.name == "test"
    assert s.feasibility_score > 0


def test_candidate_action_scorer():
    s = CandidateActionScorer()
    result = s.score({"action_type": "placement", "target_node": "node-0"})
    assert result.overall_score > 0
    assert result.explanation != ""


def test_candidate_action_scorer_batch():
    s = CandidateActionScorer()
    results = s.score_batch([
        {"action_type": "placement", "target_node": "node-0"},
        {"action_type": "scale_down", "target_node": "node-1"},
    ])
    assert len(results) == 2


def test_cost_slo_power_simulator():
    s = CostSloPowerSimulator()
    result = s.simulate_full("test")
    assert result.simulation_id != ""
    assert result.baseline_cost > 0


def test_extended_twin_service():
    s = ExtendedTwinService()
    result = s.run_comprehensive_simulation()
    assert len(result.scenarios) > 0


# ── Optimization Tests ────────────────────────────────────────

def test_elastic_worker_optimizer():
    o = ElasticWorkerOptimizer()
    cfg = o.optimize(workload={"gpu_required": 2, "framework": "pytorch"})
    assert cfg.min_workers >= 1
    assert cfg.max_workers >= cfg.min_workers


def test_elastic_worker_suggest_scale():
    o = ElasticWorkerOptimizer()
    from src.gpuopt.schemas import ElasticWorkerConfig
    cfg = ElasticWorkerConfig()
    result = o.suggest_scale(cfg, current_load=0.9)
    assert "action" in result


def test_gpu_tier_selector():
    s = GpuTierSelector()
    tiers = s.list_available_tiers()
    assert len(tiers) > 0
    result = s.select(current_gpu_model="H100", workload_profile={"memory_required_gb": 40})
    assert result.recommended_gpu_model != ""


def test_consolidation_planner():
    p = ConsolidationPlanner()
    plan = p.plan("test")
    assert plan.current_node_count > 0


def test_recommendation_prioritizer():
    p = RecommendationPrioritizer()
    result = p.prioritize([{"severity": "high", "id": 1}, {"severity": "low", "id": 2}])
    assert len(result) == 2
    assert result[0].priority_score >= result[1].priority_score


# ── Training Tests ────────────────────────────────────────────

def test_queue_aware_placement():
    p = QueueAwarePlacementEngine()
    result = p.place({"gpu_required": 2, "memory_required_gb": 16, "priority": 5})
    assert result.job_id != ""
    assert result.placement_node != ""


def test_elastic_scaling_planner():
    p = ElasticScalingPlanner()
    plan = p.plan_scale(workload={"framework": "jax"}, current_workers=4, cluster_load=0.5)
    assert plan.target_workers >= 1


def test_checkpoint_manager():
    m = CheckpointManager()
    cfg = m.configure(job_params={"max_duration_minutes": 240, "model_size_gb": 7})
    assert cfg.checkpoint_interval_minutes > 0
    assert cfg.checkpoint_size_gb > 0


def test_heterogeneous_gpu_assigner():
    a = HeterogeneousGpuAssigner()
    result = a.assign(available_gpus=[
        {"model": "H100", "memory_mb": 81920, "compute_cap": "9.0"},
        {"model": "A100", "memory_mb": 81920, "compute_cap": "8.0"},
    ])
    assert result.primary_gpu_model != ""


def test_hpo_manager():
    m = HPOManager()
    job = m.create_job()
    assert job.status == "running"
    trial = m.submit_trial(job.job_id, {"lr": 0.001})
    assert "trial_id" in trial
    m.complete_trial(job.job_id, trial["trial_id"], 0.5)
    cfg = m.get_best_config(job.job_id)
    assert "best_hyperparameters" in cfg


def test_extended_training_service():
    s = ExtendedTrainingService()
    result = s.submit_training_job({"name": "test", "gpu_required": 2, "memory_required_gb": 16})
    assert result["status"] == "planned"


# ── Inference Tests ───────────────────────────────────────────

def test_replica_right_sizer():
    s = ReplicaRightSizer()
    result = s.right_size("llama", current_replicas=4, current_latency_p99=200, target_latency_p99=100)
    assert result.recommended_replicas > 0


def test_slo_aware_scaler():
    s = SloAwareScaler()
    policy = s.create_policy("test")
    assert policy.min_replicas == 1
    result = s.evaluate_scale(policy, current_load_tps=500)
    assert "action" in result


def test_model_instance_placer():
    p = ModelInstancePlacer()
    placements = p.place("test-model", num_replicas=2)
    assert len(placements) > 0
    assert placements[0].gpu_memory_allocated_gb > 0


def test_routing_recommender():
    r = RoutingRecommender()
    result = r.recommend("test-model")
    assert result.recommended_routing != ""


def test_moe_optimizer():
    m = MoEOptimizer()
    cfg = m.configure(8, 2)
    assert cfg.num_experts == 8
    assert cfg.top_k == 2
    expert_cfg = m.recommend_expert_allocation(70, 640)
    assert expert_cfg["num_experts"] >= 4


# ── Governance Tests ──────────────────────────────────────────

def test_policy_enforcer():
    e = PolicyEnforcer()
    env = e.create_envelope("test-policy", "compute", [{"field": "gpu_count", "operator": "lte", "value": 8}])
    assert env.name == "test-policy"
    result = e.evaluate(env, {"gpu_count": 4})
    assert result["passed"] is True
    result = e.evaluate(env, {"gpu_count": 16})
    assert result["passed"] is False


def test_rollback_manager():
    m = RollbackManager()
    plan = m.create_plan("action-1", "scale_down", "Test rollback")
    assert plan.steps is not None
    assert len(plan.steps) > 0
    result = m.execute(plan)
    assert result["status"] == "completed"


def test_tenant_quota_manager():
    m = TenantQuotaManager()
    q = m.set_quota("tenant-1", "Tenant One", max_gpus=8, max_memory_gb=160, priority=5)
    assert q.max_gpus == 8
    check = m.check_quota("tenant-1", 4, 80)
    assert check["allowed"] is True
    m.update_usage("tenant-1", 6, 120)
    assert m.get_quota("tenant-1") is not None


def test_explanation_generator():
    g = ExplanationGenerator()
    exp = g.generate("placement", "job-1", {"node": "gpu-node-0", "mem": 32, "util": 45})
    assert exp.summary != ""
    assert len(exp.factors) > 0


def test_report_generator():
    g = ReportGenerator()
    report = g.generate_compliance_report("cluster-1")
    assert report.report_type == "compliance"
    assert report.tenant_summaries is not None


# ── Integration Tests ─────────────────────────────────────────

def test_prometheus_integrator():
    p = PrometheusIntegrator()
    target = p.register_target("http://prometheus:9090")
    assert target.endpoint == "http://prometheus:9090"
    metrics = p.get_metric(target, "gpu_utilization")
    assert len(metrics) > 0


def test_opentelemetry_integrator():
    o = OpenTelemetryIntegrator()
    cfg = o.configure("gpuopt-test", "localhost:4317")
    assert cfg.service_name == "gpuopt-test"
    span = o.create_span("test-span")
    assert "span_id" in span
    metric = o.record_metric("test_metric", 42.0)
    assert metric["name"] == "test_metric"


def test_ai_runtime_detector():
    d = AiRuntimeDetector()
    runtimes = d.detect()
    assert len(runtimes) >= 4
    assert runtimes[0].runtime_type == "pytorch"


def test_object_store_connector():
    s = ObjectStoreConnector()
    cfg = s.configure("s3", "s3.amazonaws.com", "my-bucket", "us-east-1",
                       {"access_key": "test", "secret_key": "test"})
    assert cfg.store_type == "s3"
    buckets = s.list_buckets(cfg)
    assert len(buckets) > 0
    assert s.upload_file(cfg, "test.txt", b"hello")
    data = s.download_file(cfg, "test.txt")
    assert data == b"mock file content"


def test_integration_manager():
    m = IntegrationManager()
    statuses = m.check_all_integrations()
    assert len(statuses) > 0
    health = m.health()
    assert "status" in health
