from __future__ import annotations

from gpuopt.predictor.ensemble_failure_predictor import EnsembleFailurePredictor
from gpuopt.ml.engine import MLEngine


# ── Ensemble Failure Predictor ───────────────────────────────

def test_predictor_initial_state():
    import tempfile
    p = EnsembleFailurePredictor(tempfile.mkdtemp())
    assert p.is_trained is False
    assert p.feature_names == []
    assert p.VERSION == "2.0.0"


def test_predictor_feature_extraction():
    p = EnsembleFailurePredictor()
    telemetry = {
        "gpu_utilization": 75.0, "memory_utilization": 80.0, "temperature": 70.0,
        "power_usage": 300.0, "clock_speed": 1500.0, "ecc_errors": 5, "retired_pages": 1,
        "xid_errors": 2, "utilization_variance": 0.3, "temperature_variance": 0.2,
        "available_gpus": 6, "total_gpus": 8, "queue_length": 10, "job_failures": 2,
        "job_retries": 1, "average_job_duration": 3600,
    }
    feats = p.extract_features(telemetry)
    assert len(feats) == 26
    assert feats[0] == 75.0
    assert feats[16] == 0.75  # gpu_util_ratio
    assert feats[24] > 0  # stress_index


def test_predictor_train():
    p = EnsembleFailurePredictor()
    result = p.train(n_synthetic=500)
    assert result["status"] == "training_complete"
    assert p.is_trained is True
    assert p.feature_count == 26
    assert "metrics" in result
    assert result["metrics"]["accuracy"] > 0
    assert result["metrics"]["roc_auc"] > 0
    assert result["metrics"]["f1_score"] > 0
    assert len(result["feature_importance"]) > 0


def test_predictor_predict_before_train():
    p = EnsembleFailurePredictor()
    result = p.predict_failure({"gpu_utilization": 50, "temperature": 60})
    assert "version" in result
    assert result["probability_raw"] >= 0


def test_predictor_predict_after_train():
    p = EnsembleFailurePredictor()
    p.train(n_synthetic=500)
    result = p.predict_failure({
        "gpu_utilization": 95.0, "memory_utilization": 95.0, "temperature": 88.0,
        "power_usage": 400.0, "clock_speed": 1800.0, "ecc_errors": 15, "retired_pages": 3,
        "xid_errors": 8, "utilization_variance": 0.5, "temperature_variance": 0.4,
        "available_gpus": 2, "total_gpus": 8, "queue_length": 20, "job_failures": 5,
        "job_retries": 3, "average_job_duration": 600,
    })
    assert "probability_raw" in result
    assert "probability_calibrated" in result
    assert "model_probs" in result
    assert "random_forest" in result["model_probs"]
    assert "gradient_boosting" in result["model_probs"]
    assert "neural_network" in result["model_probs"]
    assert len(result["risk_factors"]) > 0
    assert result["is_anomaly"] is not None


def test_predictor_analyze_cluster():
    p = EnsembleFailurePredictor()
    p.train(n_synthetic=500)
    result = p.analyze_cluster("test-cluster", node_count=4)
    assert result["status"] == "success"
    assert len(result["nodes"]) == 4
    assert "high_risk_nodes" in result["summary"]
    assert "cluster_health" in result


def test_predictor_feature_importance():
    p = EnsembleFailurePredictor()
    p.train(n_synthetic=500)
    fi = p.get_feature_importance()
    assert "random_forest_top10" in fi
    assert "gradient_boosting_top10" in fi
    assert len(fi["random_forest_top10"]) == 10


def test_predictor_model_info():
    p = EnsembleFailurePredictor()
    p.train(n_synthetic=500)
    info = p.get_model_info()
    assert info["is_trained"] is True
    assert info["version"] == "2.0.0"
    assert "random_forest" in info["models"]


def test_predictor_synthetic_data_generation():
    p = EnsembleFailurePredictor()
    X, y = p.generate_synthetic_data(200)
    assert X.shape[0] == 200
    assert X.shape[1] == 26
    assert y.shape[0] == 200
    assert set(y).issubset({0, 1})


# ── ML Engine Integration ────────────────────────────────────

def test_ml_engine_health():
    engine = MLEngine()
    h = engine.health()
    assert h["status"] == "healthy"
    assert h["ensemble_trained"] is not None
    assert "registry" in h
    assert "automl" in h
    assert "twin_sim" in h


def test_ml_engine_train_and_predict():
    engine = MLEngine()
    result = engine.train_ensemble(n_synthetic=500)
    assert result["status"] == "training_complete"
    assert "registry_entry" in result

    pred = engine.predict_failure({
        "gpu_utilization": 90.0, "memory_utilization": 92.0, "temperature": 82.0,
        "power_usage": 380.0, "clock_speed": 1600.0, "ecc_errors": 10,
        "xid_errors": 5, "utilization_variance": 0.4,
        "available_gpus": 4, "total_gpus": 8, "queue_length": 15,
        "job_failures": 3, "job_retries": 2, "average_job_duration": 1800,
        "retired_pages": 2, "temperature_variance": 0.3,
    })
    assert "probability_raw" in pred
    assert "model_probs" in pred


def test_ml_engine_analyze_cluster():
    engine = MLEngine()
    engine.train_ensemble(n_synthetic=300)
    result = engine.analyze_cluster("prod-cluster", node_count=4)
    assert len(result["nodes"]) == 4


def test_ml_engine_model_info():
    engine = MLEngine()
    engine.train_ensemble(n_synthetic=300)
    info = engine.get_model_info()
    assert info["is_trained"] is True


def test_ml_engine_feature_importance():
    engine = MLEngine()
    engine.train_ensemble(n_synthetic=300)
    fi = engine.get_feature_importance()
    assert "ensemble_mean" in fi


def test_ml_engine_registry():
    engine = MLEngine()
    engine.train_ensemble(n_synthetic=300)
    models = engine.registry_list()
    assert "ensemble_failure_predictor" in models
    info = engine.registry_get("ensemble_failure_predictor")
    assert info is not None
    assert info["version"] == "2.0.0"


def test_ml_engine_registry_promote():
    engine = MLEngine()
    engine.train_ensemble(n_synthetic=300)
    result = engine.registry_promote("ensemble_failure_predictor", "2.0.0", "production")
    assert result is not None
    assert result["new_production"]["status"] == "production"


# ── Digital Twin Simulation ──────────────────────────────────

def test_simulate_default():
    engine = MLEngine()
    result = engine.simulate()
    assert "simulation_id" in result
    assert len(result["current_state"]) == 8
    assert "aggregate" in result
    assert result["aggregate"]["avg_util"] > 0


def test_simulate_custom():
    engine = MLEngine()
    result = engine.simulate(num_gpus=4, gpu_model="NVIDIA A100-SXM-40GB",
                              workload_type="llm_training", duration_steps=30)
    assert len(result["current_state"]) == 4
    assert "tensor_activity_pct" in result["current_state"][0]


def test_simulate_failure_thermal():
    engine = MLEngine()
    result = engine.simulate_failure("thermal_runaway", num_gpus=4)
    assert result["scenario"] == "thermal_runaway"
    assert result["num_gpus"] == 4
    assert "failure_detected" in result
    assert "timeline" in result
    assert "root_cause" in result


def test_simulate_failure_memory():
    engine = MLEngine()
    result = engine.simulate_failure("memory_leak")
    assert result["scenario"] == "memory_leak"


def test_workload_profiles():
    engine = MLEngine()
    profiles = engine.list_profiles()
    assert len(profiles) == 6
    names = [p["name"] for p in profiles]
    assert "llm_inference" in names
    assert "llm_training" in names
    assert "cnn_training" in names


# ── AutoML ───────────────────────────────────────────────────

def test_automl_compare_models():
    engine = MLEngine()
    results = engine.automl_compare_models(n_samples=300)
    assert len(results) >= 2
    assert results[0]["roc_auc_mean"] > 0


# ── API Tests ────────────────────────────────────────────────

def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/ml/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


def test_train_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/ml/train", json={"n_synthetic": 200})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "training_complete"


def test_predict_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        client.post("/api/v1/ml/train", json={"n_synthetic": 200})
        r = client.post("/api/v1/ml/predict", json={
            "gpu_utilization": 85.0, "memory_utilization": 90.0, "temperature": 75.0,
        })
        assert r.status_code == 200
        assert "model_probs" in r.json()


def test_analyze_cluster_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        client.post("/api/v1/ml/train", json={"n_synthetic": 200})
        r = client.post("/api/v1/ml/analyze-cluster?cluster_id=test&node_count=3")
        assert r.status_code == 200
        assert len(r.json()["nodes"]) == 3


def test_simulate_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/ml/simulate?num_gpus=4&duration_steps=10")
        assert r.status_code == 200
        assert len(r.json()["current_state"]) == 4


def test_simulate_failure_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/ml/simulate-failure?scenario=power_spike&num_gpus=4")
        assert r.status_code == 200
        assert r.json()["scenario"] == "power_spike"


def test_workload_profiles_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/ml/workload-profiles")
        assert r.status_code == 200
        assert len(r.json()) == 6


def test_model_info_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        client.post("/api/v1/ml/train", json={"n_synthetic": 200})
        r = client.get("/api/v1/ml/model-info")
        assert r.status_code == 200
        assert r.json()["is_trained"] is True


def test_automl_compare_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/ml/automl/compare-models?n_samples=200")
        assert r.status_code == 200
        assert len(r.json()) >= 2


def test_registry_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        client.post("/api/v1/ml/train", json={"n_synthetic": 200})
        r = client.get("/api/v1/ml/registry")
        assert r.status_code == 200
        assert "ensemble_failure_predictor" in r.json()
