from __future__ import annotations

from uuid import uuid4

from gpuopt.schemas import CloudProvider, GpuPricingTier
from gpuopt.finops import FinOpsService


def test_get_pricing_all():
    pricing = FinOpsService.get_pricing()
    assert len(pricing) > 10


def test_get_pricing_filter_gpu():
    pricing = FinOpsService.get_pricing(gpu_model="h100")
    assert all(r.gpu_model == "h100" for r in pricing)
    assert len(pricing) >= 4


def test_get_pricing_filter_provider():
    pricing = FinOpsService.get_pricing(provider=CloudProvider.GCP)
    assert all(r.provider == CloudProvider.GCP for r in pricing)


def test_get_pricing_filter_tier():
    pricing = FinOpsService.get_pricing(tier=GpuPricingTier.SPOT)
    assert all(r.tier == GpuPricingTier.SPOT for r in pricing)


def test_compare_providers_default():
    comp = FinOpsService.compare_providers()
    assert comp.gpu_model == "h100"
    assert comp.gpu_count == 8
    assert len(comp.providers) > 0
    assert comp.cheapest_ondemand is not None
    assert comp.cheapest_overall is not None
    assert comp.recommendation != ""


def test_compare_providers_a100():
    comp = FinOpsService.compare_providers(gpu_model="a100", gpu_count=16)
    assert comp.gpu_model == "a100"
    assert comp.gpu_count == 16
    assert len(comp.providers) >= 1
    assert comp.max_potential_savings_percent >= 0


def test_compare_providers_single_gpu():
    comp = FinOpsService.compare_providers(gpu_model="t4", gpu_count=1)
    assert comp.gpu_model == "t4"
    assert len(comp.providers) >= 1


def _make_cluster(client, name: str):
    import uuid
    r = client.post("/api/v1/clusters", json={
        "name": f"{name}-{uuid.uuid4().hex[:8]}",
        "environment": "sandbox",
        "connector_type": "mock",
    })
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    client.post(f"/api/v1/clusters/{cid}/state")
    return cid


def test_spot_savings():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_spot.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        cid = _make_cluster(client, "spot")
        svc = FinOpsService(get_repository())
        spot = svc.analyze_spot_savings(cid)
        assert spot.total_gpus >= 0
        assert spot.savings_percent >= 0
        assert len(spot.recommendations) > 0


def test_reserved_instances():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_ri.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        cid = _make_cluster(client, "ri")
        svc = FinOpsService(get_repository())
        ri = svc.recommend_reserved_instances(cid)
        assert ri.current_monthly_cost >= 0
        assert ri.monthly_savings_1yr >= 0
        assert ri.annual_savings_1yr >= 0
        assert len(ri.recommendations) > 0


def test_budget_alert():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_budget.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        cid = _make_cluster(client, "budget")
        svc = FinOpsService(get_repository())
        alert = svc.get_budget_alert(cid, monthly_budget=100.0)
        assert alert.budget_utilization_percent >= 0
        assert alert.status in ("on_track", "watch", "at_risk", "over_budget")


def test_cost_forecast():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_forecast.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        cid = _make_cluster(client, "forecast")
        svc = FinOpsService(get_repository())
        forecast = svc.forecast_cost(cid, months=6, growth_rate=0.03)
        assert len(forecast.forecast) == 6
        assert forecast.current_monthly_cost >= 0
        assert forecast.growth_rate == 0.03


def test_cost_forecast_no_cluster():
    fake_id = uuid4()
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_forecast_nc.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    svc = FinOpsService(get_repository())
    forecast = svc.forecast_cost(fake_id, months=3)
    assert len(forecast.forecast) == 3


def test_what_if_cost():
    scenario = FinOpsService.what_if_cost(
        scenario_name="Switch to spot",
        description="Move all GPU workloads to spot instances",
        current_monthly_cost=50000.0,
        gpu_count_change=0,
        tier_change=GpuPricingTier.SPOT,
    )
    assert scenario.scenario_name == "Switch to spot"
    assert scenario.scenario_monthly_cost < scenario.current_monthly_cost
    assert scenario.annual_difference < 0
    assert len(scenario.recommendations) > 0


def test_what_if_cost_add_gpus():
    scenario = FinOpsService.what_if_cost(
        scenario_name="Scale up",
        description="Add 8 more GPUs",
        current_monthly_cost=50000.0,
        gpu_count_change=8,
    )
    assert scenario.scenario_monthly_cost > scenario.current_monthly_cost


def test_what_if_cost_equinix():
    scenario = FinOpsService.what_if_cost(
        scenario_name="Move to bare metal",
        current_monthly_cost=50000.0,
        provider_change=CloudProvider.EQUINIX,
    )
    assert scenario.scenario_monthly_cost < scenario.current_monthly_cost


def test_cost_allocation():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_alloc.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        cid = _make_cluster(client, "alloc")
        svc = FinOpsService(get_repository())
        tags = svc.get_cost_allocation(cid)
        assert isinstance(tags, list)


def test_generate_finops_recommendations():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_recs.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        cid = _make_cluster(client, "recs")
        client.post(f"/api/v1/clusters/{cid}/state")
        svc = FinOpsService(get_repository())
        recs = svc.generate_finops_recommendations(cid)
        assert isinstance(recs, list)


def test_finops_aggregate():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_agg.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        _make_cluster(client, "agg")
        svc = FinOpsService(get_repository())
        agg = svc.aggregate_costs()
        assert agg.cluster_count >= 1
        assert agg.total_gpus >= 0
        assert agg.total_monthly_cost >= 0


def test_finops_api_endpoints():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_finops_api.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_finops_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_finops_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/api/v1/finops/pricing")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 10

        r = client.get("/api/v1/finops/pricing?gpu_model=h100&provider=aws")
        assert r.status_code == 200
        assert len(r.json()) >= 1

        r = client.get("/api/v1/finops/compare?gpu_model=h100&gpu_count=8")
        assert r.status_code == 200
        comp = r.json()
        assert comp["gpu_model"] == "h100"
        assert comp["gpu_count"] == 8

        r = client.get("/api/v1/finops/compare?gpu_model=a100&gpu_count=16")
        assert r.status_code == 200

        cid = _make_cluster(client, "api-test")

        r = client.get(f"/api/v1/finops/spot-savings/{cid}")
        assert r.status_code == 200
        spot = r.json()
        assert "total_gpus" in spot
        assert "annual_savings" in spot

        r = client.get(f"/api/v1/finops/reserved-recs/{cid}")
        assert r.status_code == 200
        ri = r.json()
        assert "annual_savings_1yr" in ri

        r = client.get(f"/api/v1/finops/budget/{cid}?monthly_budget=50000")
        assert r.status_code == 200
        budget = r.json()
        assert "status" in budget
        assert "budget_utilization_percent" in budget

        r = client.get(f"/api/v1/finops/forecast/{cid}?months=6&growth_rate=0.03")
        assert r.status_code == 200
        forecast = r.json()
        assert len(forecast["forecast"]) == 6

        r = client.get("/api/v1/finops/aggregate")
        assert r.status_code == 200
        agg = r.json()
        assert "cluster_count" in agg
        assert agg["cluster_count"] >= 1

        r = client.post("/api/v1/finops/what-if?scenario_name=Switch%20to%20spot&current_monthly_cost=50000&tier=spot")
        assert r.status_code == 200
        sc = r.json()
        assert sc["scenario_monthly_cost"] < sc["current_monthly_cost"]

        r = client.get(f"/api/v1/finops/allocation/{cid}")
        assert r.status_code == 200
        tags = r.json()
        assert isinstance(tags, list)

        r = client.get(f"/api/v1/finops/recommendations/{cid}")
        assert r.status_code == 200
        recs = r.json()
        assert isinstance(recs, list)
