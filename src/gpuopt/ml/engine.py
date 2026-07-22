from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..gpu_catalog import get_gpu_catalog
from .automl import AutoMLEngine
from .closed_loop import ClosedLoopTrainer
from .cluster_algorithm import ClusterManagementAlgorithm
from .digital_twin_sim import DigitalTwinSimulationService
from .model_registry import ModelRegistry
from .node_simulation import ClusterTopology, EnhancedSimulationEngine
from .training_data_pipeline import TrainingDataCollector, collect_and_train
from .web_datasets import WebDatasetIngestion

logger = logging.getLogger(__name__)


class MLEngine:
    def __init__(self) -> None:
        from ..config import get_settings
        settings = get_settings()
        base = Path(settings.database_path).parent

        self.registry = ModelRegistry(base / "models" / "registry")
        self.automl = AutoMLEngine()
        self.twin_sim = DigitalTwinSimulationService()

        self._ensemble_predictor: Any = None
        self._data_collector: TrainingDataCollector | None = None
        self._web_ingestion: WebDatasetIngestion | None = None

    @property
    def ensemble_predictor(self) -> Any:
        if self._ensemble_predictor is None:
            from ..predictor.ensemble_failure_predictor import EnsembleFailurePredictor
            self._ensemble_predictor = EnsembleFailurePredictor()
        return self._ensemble_predictor

    @property
    def data_collector(self) -> TrainingDataCollector:
        if self._data_collector is None:
            from ..gpu_monitor import GPUMonitor
            from ..repository import ClusterRepository
            from ..config import get_settings
            repo = ClusterRepository(get_settings().database_path)
            self._data_collector = TrainingDataCollector(
                repository=repo,
                gpu_monitor=GPUMonitor(),
            )
        return self._data_collector

    def predict_failure(self, telemetry: dict) -> dict:
        return self.ensemble_predictor.predict_failure(telemetry)

    def train_ensemble(self, telemetry_history: list[dict] | None = None,
                       labels: list[int] | None = None, n_synthetic: int = 2000) -> dict:
        result = self.ensemble_predictor.train(telemetry_history, labels, n_synthetic)

        self.registry.register_model(
            name="ensemble_failure_predictor",
            version=self.ensemble_predictor.VERSION,
            framework="scikit-learn",
            metrics=self.ensemble_predictor.training_metrics,
            params={
                "n_estimators_rf": 200,
                "n_estimators_gb": 150,
                "nn_layers": [64, 32, 16],
                "n_synthetic": n_synthetic,
            },
            description="Ensemble: RF + GB + NN with IsolationForest anomaly detection",
        )

        result["registry_entry"] = f"ensemble_failure_predictor v{self.ensemble_predictor.VERSION}"
        return result

    def train_on_cluster_data(
        self, max_samples: int = 500, n_synthetic: int = 1000
    ) -> dict:
        collector = self.data_collector
        telemetry_data, labels = collector.build_training_dataset(max_samples)
        real_count = len(telemetry_data)
        if real_count == 0:
            logger.info("No real cluster data available, training on synthetic only")
            return self.train_ensemble(
                telemetry_history=None, labels=None, n_synthetic=n_synthetic + 1000,
            )
        result = self.train_ensemble(
            telemetry_history=telemetry_data,
            labels=labels,
            n_synthetic=max(n_synthetic, 2000 - real_count),
        )
        result["real_samples"] = real_count
        result["source"] = "cluster_management_data"
        return result

    def analyze_cluster(self, cluster_id: str, node_count: int = 8) -> dict:
        return self.ensemble_predictor.analyze_cluster(cluster_id, node_count)

    def get_model_info(self) -> dict:
        return self.ensemble_predictor.get_model_info()

    def get_feature_importance(self) -> dict:
        return self.ensemble_predictor.get_feature_importance()

    def simulate(self, num_gpus: int = 8, gpu_model: str = "NVIDIA H100-SXM-80GB",
                 workload_type: str = "llm_inference", duration_steps: int = 60) -> dict:
        return self.twin_sim.simulate(num_gpus, gpu_model, workload_type, duration_steps)

    def simulate_failure(self, scenario: str = "thermal_runaway", num_gpus: int = 8) -> dict:
        return self.twin_sim.simulate_failure(scenario, num_gpus)

    def list_profiles(self) -> list[dict]:
        return self.twin_sim.list_profiles()

    def automl_random_search(self, model_type: str, n_iter: int = 20,
                              n_samples: int = 1000, cv_folds: int = 5) -> dict:
        X, y = self.ensemble_predictor.generate_synthetic_data(n_samples)
        return self.automl.random_search(model_type, X, y, n_iter, cv_folds)

    def automl_compare_models(self, n_samples: int = 1000) -> list[dict]:
        X, y = self.ensemble_predictor.generate_synthetic_data(n_samples)
        return self.automl.compare_all_models(X, y)

    def registry_list(self) -> dict:
        return self.registry.list_models()

    def registry_get(self, name: str, version: str | None = None) -> dict | None:
        return self.registry.get_model(name, version)

    def registry_promote(self, name: str, version: str, stage: str = "production") -> dict | None:
        if stage == "production":
            return self.registry.promote_to_production(name, version)
        return self.registry.promote_to_staging(name, version)

    @property
    def web_ingestion(self) -> WebDatasetIngestion:
        if self._web_ingestion is None:
            self._web_ingestion = WebDatasetIngestion()
        return self._web_ingestion

    def list_web_datasets(self) -> list[dict]:
        return self.web_ingestion.list_datasets()

    def download_web_dataset(self, name: str, force: bool = False) -> dict:
        from .training_data_pipeline import TrainingDataCollector
        path = self.web_ingestion.download_dataset(name, force=force)
        data = self.web_ingestion.ingest(name)
        labels = [TrainingDataCollector.generate_labels(t) for t in data]
        pos_pct = 100 * sum(labels) / max(len(labels), 1)
        return {
            "source": name,
            "path": str(path),
            "samples": len(data),
            "positive_samples": sum(labels),
            "positive_pct": round(pos_pct, 1),
            "feature_columns": list(TrainingDataCollector.FEATURE_NAMES),
        }

    def train_on_web_data(
        self,
        sources: list[str] | None = None,
        max_samples: int = 5000,
        blend_with_cluster: bool = True,
        synthetic_factor: float = 0.5,
    ) -> dict:
        from .training_data_pipeline import TrainingDataCollector
        web_data, web_labels = self.web_ingestion.get_training_data(
            sources=sources, max_samples=max_samples,
        )
        cluster_data: list[dict] = []
        cluster_labels: list[int] = []
        if blend_with_cluster:
            collector = self.data_collector
            cluster_data, cluster_labels = collector.build_training_dataset(max_samples=500)
        combined_data = web_data + cluster_data
        combined_labels = web_labels + cluster_labels
        if not combined_data:
            logger.warning("No data available from any source, using synthetic data")
            return self.train_ensemble(
                telemetry_history=None, labels=None,
                n_synthetic=int(max_samples * synthetic_factor) + 1000,
            )
        n_synth = int(len(combined_data) * synthetic_factor)
        result = self.train_ensemble(
            telemetry_history=combined_data, labels=combined_labels,
            n_synthetic=max(n_synth, 1000),
        )
        result["web_samples"] = len(web_data)
        result["cluster_samples"] = len(cluster_data)
        result["total_samples"] = len(combined_data)
        result["source"] = "web_datasets"
        return result

    @property
    def cluster_manager(self) -> ClusterManagementAlgorithm:
        return ClusterManagementAlgorithm(predictor=self.ensemble_predictor)

    @property
    def closed_loop_trainer(self) -> ClosedLoopTrainer:
        return ClosedLoopTrainer(
            engine=self.ensemble_predictor,
            management_algorithm=self.cluster_manager,
        )

    def enhanced_simulate(
        self,
        gpu_model: str = "NVIDIA H100-SXM-80GB",
        num_gpus: int = 8,
        num_nodes: int = 1,
        steps: int = 100,
        workload_type: str = "llm_training",
    ) -> dict:
        profile_map = {
            "llm_inference": {"gpu_util_target": 45.0, "memory_target_pct": 40.0, "tensor_intensity": 0.7, "mem_intensity": 0.5},
            "llm_training": {"gpu_util_target": 92.0, "memory_target_pct": 90.0, "tensor_intensity": 0.95, "mem_intensity": 0.8},
            "cnn_training": {"gpu_util_target": 88.0, "memory_target_pct": 50.0, "tensor_intensity": 0.85, "mem_intensity": 0.4},
            "batch_inference": {"gpu_util_target": 75.0, "memory_target_pct": 30.0, "tensor_intensity": 0.5, "mem_intensity": 0.3},
            "data_processing": {"gpu_util_target": 35.0, "memory_target_pct": 20.0, "tensor_intensity": 0.2, "mem_intensity": 0.7},
        }
        profile = profile_map.get(workload_type, profile_map["llm_inference"])
        engine = EnhancedSimulationEngine()
        if "H100" in gpu_model or "dgx" in gpu_model.lower():
            topo = ClusterTopology().build_dgx_h100(num_nodes)
        else:
            topo = ClusterTopology().build_rtx_cluster(num_gpus)
        result = engine.simulate(topo, steps=steps, profile=profile)
        return result

    def enhanced_simulate_failure(
        self,
        scenario: str = "thermal_runaway",
        gpu_model: str = "NVIDIA H100-SXM-80GB",
        num_gpus: int = 8,
    ) -> dict:
        engine = EnhancedSimulationEngine()
        if "H100" in gpu_model:
            topo = ClusterTopology().build_dgx_h100(1)
        else:
            topo = ClusterTopology().build_rtx_cluster(num_gpus)
        return engine.simulate_failure_scenario(scenario, topo)

    def schedule_job(
        self,
        name: str = "",
        required_gpus: int = 1,
        required_memory_gib: float = 8.0,
        estimated_runtime_hours: float = 1.0,
        priority: int = 5,
        workload_type: str = "llm_inference",
        policy: str | None = None,
    ) -> dict:
        from .cluster_algorithm import JobSpec, SchedulingPolicy
        from .node_simulation import ClusterTopology
        job = JobSpec(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            name=name, required_gpus=required_gpus,
            required_memory_gib=required_memory_gib,
            estimated_runtime_hours=estimated_runtime_hours,
            priority=priority, workload_type=workload_type,
        )
        topo = ClusterTopology().build_dgx_h100(2)
        mgr = self.cluster_manager
        if policy:
            mgr.scheduling_policy = SchedulingPolicy(policy)
        decision = mgr.schedule_job(job, topo)
        return {
            "job_id": decision.job_id,
            "assigned_gpus": [f"{n}:GPU{g}" for n, g in decision.assigned_gpus],
            "policy": decision.policy.value,
            "predicted_failure_risk": round(decision.predicted_failure_risk, 4),
            "estimated_power_watts": round(decision.estimated_power_watts, 1),
            "thermal_headroom_c": round(decision.thermal_headroom_c, 1),
            "score": round(decision.score, 4),
            "rationale": decision.rationale,
        }

    def get_cluster_health(self) -> dict:
        topo = ClusterTopology().build_dgx_h100(2)
        return self.cluster_manager.get_cluster_health_report(topo)

    def closed_loop_train(
        self,
        cycles: int = 3,
        steps_per_episode: int = 80,
        retrain_every: int = 1,
        gpu_model: str = "NVIDIA H100-SXM-80GB",
        num_nodes: int = 1,
    ) -> list[dict]:
        topo = ClusterTopology().build_dgx_h100(num_nodes)
        trainer = self.closed_loop_trainer
        return trainer.iterative_improvement_cycle(topo, cycles, steps_per_episode, retrain_every)

    def compare_policies(
        self,
        steps: int = 60,
        num_nodes: int = 1,
    ) -> list[dict]:
        topo = ClusterTopology().build_dgx_h100(num_nodes)
        trainer = self.closed_loop_trainer
        return trainer.compare_policies(topo, steps=steps)

    def optimize_policies(
        self,
        iterations: int = 10,
        steps_per_eval: int = 50,
        num_nodes: int = 1,
    ) -> dict:
        topo = ClusterTopology().build_dgx_h100(num_nodes)
        trainer = self.closed_loop_trainer
        return trainer.policy_optimization_loop(topo, iterations, steps_per_eval)

    def power_cap_analysis(self) -> list[dict]:
        topo = ClusterTopology().build_dgx_h100(1)
        caps = self.cluster_manager.compute_power_caps(topo)
        return [
            {"gpu": str(c.gpu_key), "from_watts": c.current_power_watts,
             "to_watts": c.new_cap_watts, "temp_before": c.temp_before,
             "temp_after_estimate": c.temp_after_estimate, "reason": c.reason}
            for c in caps
        ]

    def drain_recommendations(self) -> list[dict]:
        topo = ClusterTopology().build_dgx_h100(1)
        drains = self.cluster_manager.recommend_drain(topo)
        return [
            {"gpu": str(d.gpu_key), "risk_score": d.risk_score,
             "urgency": d.urgency, "action": d.suggested_action, "reason": d.reason}
            for d in drains
        ]

    def list_gpu_catalog(
        self,
        vendor: str | None = None,
        segment: str | None = None,
        min_vram: float | None = None,
        capabilities: str | None = None,
    ) -> list[dict]:
        cap_list = [c.strip() for c in capabilities.split(",")] if capabilities else None
        return get_gpu_catalog().query(vendor=vendor, segment=segment, min_vram=min_vram, capabilities=cap_list)

    def lookup_gpu(self, name: str) -> dict | None:
        entry = get_gpu_catalog().lookup(name)
        return entry.to_dict() if entry else None

    def get_gpu_catalog_stats(self) -> dict:
        cat = get_gpu_catalog()
        return {
            "total_entries": len(cat.entries),
            "by_vendor": {v: len(g) for v, g in cat.group_by_vendor().items()},
            "by_segment": {s: len(g) for s, g in cat.group_by_segment().items()},
            "training_capable": len(cat.get_training_capable()),
            "inference_capable": len(cat.get_inference_capable()),
            "vendors": ["nvidia", "amd", "intel"],
            "segments": ["consumer", "workstation", "data_center", "entry"],
        }

    def schedule_job_with_capability(
        self,
        name: str = "",
        required_gpus: int = 1,
        required_memory_gib: float = 8.0,
        estimated_runtime_hours: float = 1.0,
        priority: int = 5,
        workload_type: str = "llm_inference",
        policy: str | None = None,
        required_capabilities: str | None = None,
    ) -> dict:
        from .cluster_algorithm import JobSpec, SchedulingPolicy
        from .node_simulation import ClusterTopology
        caps = [c.strip() for c in required_capabilities.split(",")] if required_capabilities else []
        job = JobSpec(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            name=name, required_gpus=required_gpus,
            required_memory_gib=required_memory_gib,
            estimated_runtime_hours=estimated_runtime_hours,
            priority=priority, workload_type=workload_type,
        )
        topo = ClusterTopology().build_dgx_h100(2)
        mgr = self.cluster_manager
        if policy:
            mgr.scheduling_policy = SchedulingPolicy(policy)
        decision = mgr.schedule_job(job, topo)

        cap_check = {}
        for cap in caps:
            cap_check[cap] = all(
                ClusterManagementAlgorithm._check_gpu_capability(
                    topo.get_gpu(0, 0).spec.model, cap,
                ) for nid, gidx in decision.assigned_gpus
            )

        return {
            "job_id": decision.job_id,
            "assigned_gpus": [f"{n}:GPU{g}" for n, g in decision.assigned_gpus],
            "policy": decision.policy.value,
            "predicted_failure_risk": round(decision.predicted_failure_risk, 4),
            "estimated_power_watts": round(decision.estimated_power_watts, 1),
            "thermal_headroom_c": round(decision.thermal_headroom_c, 1),
            "score": round(decision.score, 4),
            "rationale": decision.rationale,
            "capability_check": cap_check,
        }

    def health(self) -> dict:
        return {
            "status": "healthy",
            "ensemble_trained": self.ensemble_predictor.is_trained,
            "ensemble_version": self.ensemble_predictor.VERSION,
            "registry": self.registry.health(),
            "automl": self.automl.health(),
            "twin_sim": self.twin_sim.health(),
            "cluster_algorithm": {
                "scheduling_policies": ["round_robin", "least_loaded", "risk_aware", "thermal_aware", "power_efficient", "hybrid"],
                "power_cap_modes": ["off", "temperature_guided", "risk_guided", "predictive"],
                "failure_scenarios": ["thermal_runaway", "memory_leak", "xid_storm", "power_surge", "fan_failure"],
            },
            "closed_loop": {"available": True},
        }
