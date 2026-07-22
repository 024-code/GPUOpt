from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCloudPricingFetcher:
    def test_fallback_on_fetch_failure(self):
        from gpuopt.cloud_costs import CloudPricingFetcher, CloudPricingRow
        fetcher = CloudPricingFetcher()
        with patch.object(fetcher, "_fetch_aws", return_value=None):
            pricing = fetcher.fetch_live_pricing("aws", "us-east-1")
            assert len(pricing) > 0
            assert pricing[0].source == "static"

    def test_live_data_cached(self):
        from gpuopt.cloud_costs import CloudPricingFetcher, CloudPricingRow
        fetcher = CloudPricingFetcher()
        mock_rows = [CloudPricingRow(provider="aws", region="us-east-1", instance_type="p3.2xlarge",
                                     gpu_model="V100", gpu_count=1, hourly_cost=3.06, monthly_cost=2233.8, source="live_aws")]
        with patch.object(fetcher, "_fetch_aws", return_value=mock_rows):
            first = fetcher.fetch_live_pricing("aws")
            assert first[0].source == "live_aws"
            second = fetcher.fetch_live_pricing("aws")
            assert second[0].source == "live_aws"

    def test_fetch_aws_parses_real_data(self):
        from gpuopt.cloud_costs import CloudPricingFetcher
        fetcher = CloudPricingFetcher()
        sample = {
            "products": {
                "SKU1": {
                    "attributes": {
                        "regionCode": "us-east-1", "instanceType": "p3.2xlarge",
                        "gpuModel": "V100", "gpuCount": "1", "vcpu": "8", "memory": "61 GiB"
                    }
                }
            },
            "terms": {
                "OnDemand": {
                    "SKU1": {
                        "SKU1.T1": {
                            "priceDimensions": {
                                "D1": {"unit": "Hrs", "pricePerUnit": {"USD": "3.06"}}
                            }
                        }
                    }
                }
            }
        }
        with patch("gpuopt.cloud_costs.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = json.dumps(sample).encode()
            rows = fetcher._fetch_aws("us-east-1")
            assert rows is not None
            assert rows[0].gpu_model == "V100"
            assert rows[0].hourly_cost == 3.06
            assert rows[0].source == "live_aws"

    def test_get_pricing_falls_back_gracefully(self):
        from gpuopt.cloud_costs import CloudPricingFetcher
        fetcher = CloudPricingFetcher()
        with patch.object(fetcher, "fetch_live_pricing", return_value=[]):
            pricing = fetcher.get_pricing("aws")
            assert len(pricing) > 0

    def test_delegates_to_fallback(self):
        from gpuopt.cloud_costs import CloudPricingFetcher
        fetcher = CloudPricingFetcher()
        assert fetcher.get_all_providers() == ["aws", "azure", "gcp"]
        result = fetcher.estimate_monthly_cost("H100", 1)
        assert result > 0


class TestCostReport:
    def test_report_no_state(self, client):
        payload = {"name": "cost-no-st", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/report")
        assert resp.status_code == 404

    def test_report_with_snapshot(self, client):
        payload = {
            "name": "cost-snap",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "cost-snap"
        assert data["total_gpus"] >= 0
        assert data["total_hourly_cost"] >= 0
        assert data["total_monthly_cost"] >= 0
        assert data["efficiency_percent"] >= 0
        assert len(data["nodes"]) >= 0
        assert "summary" in data

    def test_report_with_mock_state(self, client):
        payload = {"name": "cost-mock", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gpus"] >= 0
        assert data["active_gpus"] + data["idle_gpus"] == data["total_gpus"]
        assert data["total_hourly_cost"] >= 0
        assert data["efficiency_percent"] >= 0

    def test_report_not_found(self, client):
        resp = client.get("/api/v1/clusters/00000000-0000-0000-0000-000000000000/costs/report")
        assert resp.status_code == 404


class TestSavingsProjection:
    def test_projection_no_state(self, client):
        payload = {"name": "cost-proj-no", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/projections")
        assert resp.status_code == 404

    def test_projection_with_data(self, client):
        payload = {
            "name": "cost-proj",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/projections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "cost-proj"
        assert data["current_monthly_cost"] >= 0
        assert data["projected_monthly_cost"] >= 0
        assert data["monthly_savings"] >= 0
        assert data["annual_savings"] >= 0
        assert "summary" in data

    def test_projection_not_found(self, client):
        resp = client.get("/api/v1/clusters/00000000-0000-0000-0000-000000000000/costs/projections")
        assert resp.status_code == 404


class TestCostSummary:
    def test_summary_no_state(self, client):
        payload = {"name": "cost-sum-no", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/summary")
        assert resp.status_code == 404

    def test_summary_with_data(self, client):
        payload = {
            "name": "cost-sum",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "cost-sum"
        assert data["total_gpus"] >= 0
        assert data["monthly_cost"] >= 0
        assert data["monthly_waste"] >= 0
        assert data["cost_health"] in ("good", "fair", "poor", "critical")
        assert "summary" in data

    def test_summary_with_recommendations(self, client):
        payload = {
            "name": "cost-sum-rec",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        resp = client.get(f"/api/v1/clusters/{cluster_id}/costs/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["potential_monthly_savings"] >= 0
        assert data["payback_period_days"] >= 0

    def test_summary_not_found(self, client):
        resp = client.get("/api/v1/clusters/00000000-0000-0000-0000-000000000000/costs/summary")
        assert resp.status_code == 404
