from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpuopt.predictor.failure_predictor import FailurePredictor


@pytest.fixture(autouse=True)
def reset_predictor(tmp_path: Path) -> None:
    from gpuopt.registry import get_registry, reset_registry
    import os
    os.environ.pop("GPUOPT_DATABASE_PATH", None)
    p = FailurePredictor()
    p.model_path = tmp_path / "test_model.pkl"
    p.is_trained = False
    p.load_model()
    reg = get_registry()
    reg.register("predictor", p, force=True)


def _make_telemetry(
    temp: float = 50,
    gpu_util: float = 50,
    mem_util: float = 50,
    ecc: int = 0,
    xid: int = 0,
    uv: float = 0.1,
) -> dict:
    return {
        "gpu_utilization": gpu_util,
        "memory_utilization": mem_util,
        "temperature": temp,
        "power_usage": 250,
        "clock_speed": 1500,
        "ecc_errors": ecc,
        "retired_pages": 0,
        "xid_errors": xid,
        "utilization_variance": uv,
        "temperature_variance": 0.2,
        "available_gpus": 8,
        "total_gpus": 8,
        "queue_length": 5,
        "job_failures": 0,
        "job_retries": 0,
        "average_job_duration": 300,
    }


def _generate_training_data(count: int = 150, failure_rate: float = 0.3) -> tuple[list[dict], list[int]]:
    telemetry = []
    labels = []
    for _ in range(count):
        will_fail = random.random() < failure_rate
        t = _make_telemetry(
            temp=random.uniform(70, 95) if will_fail else random.uniform(30, 60),
            gpu_util=random.uniform(70, 100) if will_fail else random.uniform(10, 60),
            mem_util=random.uniform(80, 100) if will_fail else random.uniform(20, 70),
            ecc=random.randint(5, 20) if will_fail else random.randint(0, 3),
            xid=random.randint(2, 10) if will_fail else 0,
            uv=random.uniform(0.3, 0.6) if will_fail else random.uniform(0.05, 0.2),
        )
        telemetry.append(t)
        labels.append(1 if will_fail else 0)
    return telemetry, labels


class TestFailurePredictor:
    def test_extract_features_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            features = p.extract_features(_make_telemetry())
            assert features.shape == (20,)
            assert p.feature_names == list(p.feature_names)  # list, ordered

    def test_extract_features_derived_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            t = _make_telemetry(temp=85, gpu_util=80, mem_util=90, ecc=10, xid=5)
            f = p.extract_features(t)
        expected_ratio = 80 / 100.0
        assert f[16] == pytest.approx(expected_ratio)  # gpu_utilization_ratio
        expected_temp_ratio = 85 / 85.0
        assert f[18] == pytest.approx(expected_temp_ratio)  # temperature_ratio
        expected_error_rate = (10 + 5) / 1000.0
        assert f[19] == pytest.approx(expected_error_rate)  # error_rate

    def test_predict_untrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            p.is_trained = False
            result = p.predict_failure(_make_telemetry())
            assert result["failure_predicted"] is False
            assert result["risk_factors"] == ["Model not trained yet"]

    def test_train_with_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            result = p.train([_make_telemetry()], [0])
            assert result["status"] == "insufficient_data"

    def test_train_and_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            telemetry, labels = _generate_training_data(200)
            result = p.train(telemetry, labels)
            assert result["status"] == "training_complete"
            assert result["samples"] >= 100
            assert p.is_trained is True

            healthy = _make_telemetry(temp=40, gpu_util=30, mem_util=40, ecc=0, xid=0)
            pred = p.predict_failure(healthy)
            assert "probability" in pred
            assert "risk_factors" in pred
            assert "recommendation" in pred

    def test_train_and_predict_high_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            telemetry, labels = _generate_training_data(200, failure_rate=0.4)
            p.train(telemetry, labels)

            high_risk = _make_telemetry(temp=92, gpu_util=95, mem_util=98, ecc=18, xid=8, uv=0.5)
            pred = p.predict_failure(high_risk)
            assert len(pred["risk_factors"]) >= 1

    def test_predict_identifies_risk_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            telemetry, labels = _generate_training_data(200)
            p.train(telemetry, labels)

            hot = _make_telemetry(temp=82, ecc=0, xid=0)
            pred = p.predict_failure(hot)
            assert any("temperature" in r.lower() for r in pred["risk_factors"])

            ecc_high = _make_telemetry(temp=50, ecc=15, xid=0)
            pred = p.predict_failure(ecc_high)
            assert any("ecc" in r.lower() for r in pred["risk_factors"])

            xid_high = _make_telemetry(temp=50, ecc=0, xid=7)
            pred = p.predict_failure(xid_high)
            assert any("xid" in r.lower() for r in pred["risk_factors"])

            mem_high = _make_telemetry(temp=50, ecc=0, xid=0, mem_util=95)
            pred = p.predict_failure(mem_high)
            assert any("memory" in r.lower() for r in pred["risk_factors"])

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            telemetry, labels = _generate_training_data(200)
            p.train(telemetry, labels)
            assert p.model_path.exists()

            p2 = FailurePredictor()
            p2.model_path = Path(tmp) / "test_model.pkl"
            p2.load_model()
            assert p2.is_trained is True
            assert len(p2.feature_names) == 20

    def test_analyze_cluster_untrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            result = p.analyze_cluster("c1", node_count=3)
            assert result["status"] == "success"
            assert result["cluster_id"] == "c1"
            assert len(result["nodes"]) == 3
            assert result["summary"]["total_nodes"] == 3

    def test_model_persistence_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "persist.pkl"
            p = FailurePredictor(model_path=model_file)
            telemetry, labels = _generate_training_data(200)
            p.train(telemetry, labels)
            assert model_file.exists()

            p2 = FailurePredictor(model_path=model_file)
            assert p2.is_trained is True
            result = p2.predict_failure(_make_telemetry(temp=40))
            assert "probability" in result

    def test_model_path_uses_config_by_default(self):
        p = FailurePredictor()
        from gpuopt.config import get_settings
        settings = get_settings()
        base = settings.database_path.parent
        expected = base / "models" / "failure_predictor.pkl"
        assert str(p.model_path).endswith("failure_predictor.pkl")

    def test_analyze_cluster_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = FailurePredictor()
            p.model_path = Path(tmp) / "test_model.pkl"
            telemetry, labels = _generate_training_data(200)
            p.train(telemetry, labels)
            result = p.analyze_cluster("c1", node_count=5)
            assert result["status"] == "success"
            assert len(result["nodes"]) == 5


class TestFailurePredictorAPI:
    def test_predict_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/predictor/predict", json=_make_telemetry())
        assert resp.status_code == 200
        data = resp.json()
        assert "failure_predicted" in data
        assert "probability" in data

    def test_train_endpoint_insufficient(self, client: TestClient):
        resp = client.post("/api/v1/predictor/train", json={
            "telemetry_data": [_make_telemetry()],
            "labels": [0],
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "insufficient_data"

    def test_train_endpoint_sufficient(self, client: TestClient):
        telemetry, labels = _generate_training_data(150)
        resp = client.post("/api/v1/predictor/train", json={
            "telemetry_data": telemetry,
            "labels": labels,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "training_complete"
        assert data["samples"] > 0

    def test_analyze_cluster_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/predictor/analyze-cluster?cluster_id=c1&node_count=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["nodes"]) == 3

    def test_train_then_predict(self, client: TestClient):
        telemetry, labels = _generate_training_data(150)
        client.post("/api/v1/predictor/train", json={
            "telemetry_data": telemetry,
            "labels": labels,
        })
        resp = client.post("/api/v1/predictor/predict", json=_make_telemetry())
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_factors" in data
