from __future__ import annotations

from uuid import uuid4

from gpuopt.schemas import GPU_POWER_PROFILES
from gpuopt.power import PowerService


def test_power_profiles_list():
    profiles = PowerService.list_power_profiles()
    assert len(profiles) >= 8
    models = [p["gpu_model"] for p in profiles]
    assert "h100" in models
    assert "a100" in models
    assert "t4" in models


def test_power_profile_found():
    profile = PowerService.get_power_profile("h100")
    assert profile is not None
    assert profile["tdp_watts"] == 700.0
    assert profile["idle_power_watts"] == 50.0


def test_power_profile_normalized():
    profile = PowerService.get_power_profile("NVIDIA A100")
    assert profile is not None
    assert profile["gpu_model"] == "a100"


def test_power_profile_not_found():
    profile = PowerService.get_power_profile("unknown-gpu")
    assert profile is None


def test_power_analysis():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_power_analysis.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_power_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_power_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        import uuid
        r = client.post("/api/v1/clusters", json={
            "name": f"power-analysis-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "mock",
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")

        svc = PowerService(get_repository())
        result = svc.analyze_power(cid)
        assert result.total_gpus >= 0
        assert result.total_power_draw_watts >= 0
        assert result.total_power_capacity_watts >= 0
        assert result.power_efficiency_score >= 0
        assert len(result.recommendations) > 0


def test_carbon_estimate():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_power_carbon.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_power_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_power_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        import uuid
        r = client.post("/api/v1/clusters", json={
            "name": f"power-carbon-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "mock",
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")

        svc = PowerService(get_repository())
        carbon = svc.estimate_carbon(cid)
        assert carbon.total_energy_kwh >= 0
        assert carbon.carbon_footprint_kg_co2 >= 0
        assert carbon.carbon_footprint_tons_co2 >= 0
        assert carbon.equivalent_miles_driven >= 0
        assert len(carbon.recommendations) > 0


def test_suggest_power_cap_default():
    cap = PowerService.suggest_power_cap()
    assert cap.gpu_model == "a100"
    assert cap.gpu_count == 8
    assert cap.recommended_cap_watts > 0
    assert cap.recommended_cap_percent > 0
    assert cap.estimated_performance_impact_percent >= 0
    assert len(cap.recommendations) > 0


def test_suggest_power_cap_h100():
    cap = PowerService.suggest_power_cap(gpu_model="h100", gpu_count=4, current_power_watts=600.0)
    assert cap.gpu_model == "h100"
    assert cap.gpu_count == 4
    assert cap.recommended_cap_watts < 600.0
    assert cap.estimated_power_savings_watts > 0
    assert cap.estimated_cost_savings_monthly > 0


def test_suggest_power_cap_low_power():
    cap = PowerService.suggest_power_cap(gpu_model="t4", gpu_count=2, current_power_watts=50.0)
    assert cap.current_tdp_percent < 100
    assert cap.estimated_performance_impact_percent >= 0


def test_generate_power_recommendations():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_power_recs.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_power_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_power_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        import uuid
        r = client.post("/api/v1/clusters", json={
            "name": f"power-recs-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "mock",
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/state")

        svc = PowerService(get_repository())
        recs = svc.generate_power_recommendations(cid)
        assert isinstance(recs, list)


def test_power_api_endpoints():
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = "/tmp/test_power_api.db"
    from gpuopt.config import get_settings
    from gpuopt.dependencies import get_repository, get_power_service
    get_settings.cache_clear()
    get_repository.cache_clear()
    get_power_service.cache_clear()

    from gpuopt.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/api/v1/power/profiles")
        assert r.status_code == 200
        profiles = r.json()
        assert len(profiles) >= 8

        r = client.get("/api/v1/power/profile/h100")
        assert r.status_code == 200
        assert r.json()["tdp_watts"] == 700.0

        r = client.get("/api/v1/power/profile/unknown")
        assert r.status_code == 404

        r = client.get("/api/v1/power/cap-suggestion?gpu_model=h100&gpu_count=4&current_power_watts=600")
        assert r.status_code == 200
        cap = r.json()
        assert cap["recommended_cap_watts"] > 0
        assert cap["estimated_power_savings_watts"] > 0

        r = client.get("/api/v1/power/cap-suggestion?gpu_model=t4&gpu_count=2")
        assert r.status_code == 200

        import uuid
        r = client.post("/api/v1/clusters", json={
            "name": f"power-api-{uuid.uuid4().hex[:8]}",
            "environment": "sandbox",
            "connector_type": "mock",
        })
        assert r.status_code == 201
        cid = r.json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")

        r = client.get(f"/api/v1/power/analysis/{cid}")
        assert r.status_code == 200
        analysis = r.json()
        assert "total_power_draw_watts" in analysis
        assert "power_efficiency_score" in analysis
        assert len(analysis["recommendations"]) > 0

        r = client.get(f"/api/v1/power/carbon/{cid}")
        assert r.status_code == 200
        carbon = r.json()
        assert "carbon_footprint_kg_co2" in carbon
        assert "equivalent_miles_driven" in carbon

        r = client.get(f"/api/v1/power/recommendations/{cid}")
        assert r.status_code == 200
        recs = r.json()
        assert isinstance(recs, list)
