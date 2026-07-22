from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gpuopt.ml.web_datasets import WebDatasetIngestion


@pytest.fixture
def ingestion():
    with tempfile.TemporaryDirectory() as tmp:
        yield WebDatasetIngestion(cache_dir=tmp)


class TestWebDatasetIngestion:
    def test_list_datasets(self, ingestion):
        datasets = ingestion.list_datasets()
        assert len(datasets) >= 2
        names = [d["name"] for d in datasets]
        assert "epoch_gpu_clusters" in names

    def test_download_epoch_and_parse(self, ingestion):
        path = ingestion.download_dataset("epoch_gpu_clusters")
        assert path.exists()
        assert path.stat().st_size > 100
        data = ingestion.parse_epoch_gpu_clusters(path)
        assert len(data) > 0
        keys = {"gpu_utilization", "memory_utilization", "temperature", "power_usage"}
        assert keys.issubset(data[0].keys())

    def test_ingest_epoch_gpu_clusters(self, ingestion):
        data = ingestion.ingest("epoch_gpu_clusters")
        assert len(data) > 0
        assert data[0]["total_gpus"] >= 1
        assert 0 <= data[0]["gpu_utilization"] <= 100

    def test_get_training_data(self, ingestion):
        telemetry, labels = ingestion.get_training_data(
            sources=["epoch_gpu_clusters"], max_samples=100,
        )
        assert len(telemetry) <= 100
        assert len(telemetry) == len(labels)
        assert set(labels).issubset({0, 1})

    def test_parse_custom_csv(self, ingestion):
        import csv
        path = Path(ingestion.cache_dir) / "test_custom.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gpu_util", "mem_util", "temp", "power", "ecc", "xid"])
            w.writerow(["80.5", "90.0", "82.3", "350.0", "5", "2"])
            w.writerow(["30.0", "40.0", "45.0", "100.0", "0", "0"])
        data = ingestion.parse_custom_csv(path)
        assert len(data) == 2
        assert data[0]["gpu_utilization"] == 80.5
        assert data[0]["ecc_errors"] == 5

    def test_parse_custom_csv_with_map(self, ingestion):
        import csv
        path = Path(ingestion.cache_dir) / "test_mapped.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["util_gpu", "util_mem", "temp_c", "power_w"])
            w.writerow(["95.0", "99.0", "88.0", "420.0"])
        data = ingestion.parse_custom_csv(path, column_map={
            "gpu_util": "util_gpu",
            "mem_util": "util_mem",
            "temp": "temp_c",
            "power": "power_w",
        })
        assert len(data) == 1
        assert data[0]["gpu_utilization"] == 95.0
        assert data[0]["temperature"] == 88.0

    def test_parse_custom_json(self, ingestion):
        import json
        path = Path(ingestion.cache_dir) / "test.json"
        with open(path, "w") as f:
            json.dump([
                {"gpu_utilization": 85.0, "memory_utilization": 70.0, "temperature": 65.0,
                 "power_usage": 250.0, "clock_speed": 1500.0, "ecc_errors": 3, "xid_errors": 0,
                 "total_gpus": 8, "available_gpus": 2},
            ], f)
        data = ingestion.ingest(str(path))
        assert len(data) == 1
        assert data[0]["gpu_utilization"] == 85.0

    def test_ingest_all_available(self, ingestion):
        results = ingestion.ingest_all_available()
        assert "_total_samples" in results
        assert results["_total_samples"] > 0

    def test_synthetic_fallback_no_network(self, ingestion):
        from gpuopt.ml.web_datasets import DATASET_REGISTRY
        fake_name = "fake_test_dataset"
        DATASET_REGISTRY[fake_name] = {
            "url": "https://nonexistent.example.com/data.csv",
            "description": "test",
            "license": "test",
            "telemetry": True,
        }
        try:
            path = ingestion.download_dataset(fake_name)
            assert path.exists()
            data = ingestion.parse_custom_csv(path)
            assert len(data) > 0
        finally:
            DATASET_REGISTRY.pop(fake_name, None)

    def test_get_training_data_empty_sources(self, ingestion):
        telemetry, labels = ingestion.get_training_data(sources=[], max_samples=10)
        assert telemetry == []
        assert labels == []

    def test_labels_generated(self, ingestion):
        telemetry, labels = ingestion.get_training_data(
            sources=["epoch_gpu_clusters"], max_samples=50,
        )
        for t, lbl in zip(telemetry, labels):
            from gpuopt.ml.training_data_pipeline import TrainingDataCollector
            expected = TrainingDataCollector.generate_labels(t)
            assert lbl == expected


class TestWebDatasetsEndpoints:
    def test_list_datasets_endpoint(self, client):
        resp = client.get("/api/v1/ml/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_download_dataset_endpoint(self, client):
        resp = client.post("/api/v1/ml/datasets/download", params={
            "name": "epoch_gpu_clusters",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["samples"] > 0
        assert data["source"] == "epoch_gpu_clusters"

    def test_train_on_web_datasets_endpoint(self, client):
        resp = client.post("/api/v1/ml/datasets/train", params={
            "sources": "epoch_gpu_clusters",
            "max_samples": 200,
            "blend_with_cluster": False,
            "synthetic_factor": 0.3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "training_complete"
        assert "web_samples" in data

    def test_train_on_web_data_no_sources(self, client):
        resp = client.post("/api/v1/ml/datasets/train", params={
            "max_samples": 100,
            "blend_with_cluster": False,
        })
        assert resp.status_code == 200
