from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from gpuopt.ml.training_data_pipeline import TrainingDataCollector, collect_and_train


@pytest.fixture
def collector():
    return TrainingDataCollector()


class TestTrainingDataCollector:
    def test_generate_labels_high_ecc(self, collector):
        t = {"ecc_errors": 60, "xid_errors": 0, "temperature": 50,
             "memory_utilization": 50, "gpu_utilization": 50,
             "available_gpus": 4, "total_gpus": 8, "power_usage": 200}
        assert collector.generate_labels(t) == 1

    def test_generate_labels_high_temp(self, collector):
        t = {"ecc_errors": 0, "xid_errors": 0, "temperature": 88,
             "memory_utilization": 50, "gpu_utilization": 50,
             "available_gpus": 4, "total_gpus": 8, "power_usage": 200}
        assert collector.generate_labels(t) == 1

    def test_generate_labels_high_xid(self, collector):
        t = {"ecc_errors": 0, "xid_errors": 12, "temperature": 50,
             "memory_utilization": 50, "gpu_utilization": 50,
             "available_gpus": 4, "total_gpus": 8, "power_usage": 200}
        assert collector.generate_labels(t) == 1

    def test_generate_labels_high_memory(self, collector):
        t = {"ecc_errors": 0, "xid_errors": 0, "temperature": 50,
             "memory_utilization": 97, "gpu_utilization": 50,
             "available_gpus": 4, "total_gpus": 8, "power_usage": 200}
        assert collector.generate_labels(t) == 1

    def test_generate_labels_high_power(self, collector):
        t = {"ecc_errors": 0, "xid_errors": 0, "temperature": 50,
             "memory_utilization": 50, "gpu_utilization": 50,
             "available_gpus": 4, "total_gpus": 8, "power_usage": 450}
        assert collector.generate_labels(t) == 1

    def test_generate_labels_low_risk(self, collector):
        t = {"ecc_errors": 0, "xid_errors": 0, "temperature": 50,
             "memory_utilization": 40, "gpu_utilization": 30,
             "available_gpus": 6, "total_gpus": 8, "power_usage": 150}
        assert collector.generate_labels(t) == 0

    def test_generate_labels_edge_risk(self, collector):
        t = {"ecc_errors": 5, "xid_errors": 1, "temperature": 65,
             "memory_utilization": 70, "gpu_utilization": 60,
             "available_gpus": 2, "total_gpus": 8, "power_usage": 250}
        assert collector.generate_labels(t) == 0

    def test_collect_all_empty_when_no_sources(self, collector):
        data = collector.collect_all()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_build_training_dataset_empty(self, collector):
        telemetry, labels = collector.build_training_dataset()
        assert telemetry == []
        assert labels == []

    def test_collect_from_gpu_monitor(self, collector):
        mock_snapshot = MagicMock()
        mock_snapshot.total_gpus = 4
        dev = MagicMock()
        dev.utilization_gpu_percent = 45.0
        dev.utilization_memory_percent = 60.0
        dev.temperature_celsius = 72.0
        dev.power_draw_watts = 300.0
        dev.clock_sm_mhz = 1500
        dev.ecc_errors_volatile = 3
        dev.ecc_errors_aggregate = 2
        mock_snapshot.devices = [dev]

        monitor = MagicMock()
        monitor.get_snapshot.return_value = mock_snapshot
        collector._gpu_monitor = monitor

        data = collector.collect_from_gpu_monitor()
        assert len(data) == 1
        assert data[0]["gpu_utilization"] == 45.0
        assert data[0]["ecc_errors"] == 5
        assert data[0]["temperature"] == 72.0

    def test_collect_from_gpu_monitor_no_snapshot(self, collector):
        monitor = MagicMock()
        monitor.get_snapshot.return_value = None
        collector._gpu_monitor = monitor
        data = collector.collect_from_gpu_monitor()
        assert len(data) == 0

    def test_build_training_dataset_with_mock_monitor(self, collector):
        mock_snapshot = MagicMock()
        mock_snapshot.total_gpus = 4
        dev = MagicMock()
        dev.utilization_gpu_percent = 90.0
        dev.utilization_memory_percent = 95.0
        dev.temperature_celsius = 88.0
        dev.power_draw_watts = 400.0
        dev.clock_sm_mhz = 1800
        dev.ecc_errors_volatile = 20
        dev.ecc_errors_aggregate = 10
        mock_snapshot.devices = [dev]

        monitor = MagicMock()
        monitor.get_snapshot.return_value = mock_snapshot
        collector._gpu_monitor = monitor

        telemetry, labels = collector.build_training_dataset()
        assert len(telemetry) == 1
        assert len(labels) == 1
        assert labels[0] == 1


class TestCollectAndTrain:
    def test_collect_and_train_with_empty_data(self):
        mock_engine = MagicMock()
        mock_engine.train_ensemble.return_value = {"status": "training_complete"}

        result = collect_and_train(mock_engine, max_samples=100, n_synthetic=500)
        assert result["status"] == "training_complete"
        mock_engine.train_ensemble.assert_called_once()

    def test_collect_and_train_passes_real_data(self):
        mock_engine = MagicMock()
        mock_engine.train_ensemble.return_value = {"status": "training_complete"}
        mock_repo = MagicMock()

        with patch(
            "gpuopt.ml.training_data_pipeline.TrainingDataCollector"
        ) as MockCollector:
            instance = MockCollector.return_value
            instance.build_training_dataset.return_value = (
                [{"gpu_utilization": 90}],
                [1],
            )
            result = collect_and_train(
                mock_engine, max_samples=100, n_synthetic=500,
                repository=mock_repo,
                include_web_datasets=False,
            )
            assert result["status"] == "training_complete"
            call_kwargs = mock_engine.train_ensemble.call_args[1]
            assert call_kwargs["telemetry_history"] == [{"gpu_utilization": 90}]
            assert call_kwargs["labels"] == [1]


    def test_data_collection_status_endpoint(self, client):
        resp = client.get("/api/v1/ml/data-collection-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "sources" in data

    def test_train_from_cluster_endpoint(self, client):
        resp = client.post("/api/v1/ml/train-from-cluster", params={
            "max_samples": 100, "n_synthetic": 500,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "training_complete"
