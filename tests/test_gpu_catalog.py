from __future__ import annotations

import pytest

from gpuopt.gpu_catalog import GpuCatalog, get_gpu_catalog, lookup_gpu


@pytest.fixture
def catalog():
    return get_gpu_catalog()


class TestGpuCatalog:
    def test_contains_all_vendors(self, catalog):
        entries = catalog.entries
        vendors = {e.vendor.value for e in entries}
        assert "nvidia" in vendors
        assert "amd" in vendors
        assert "intel" in vendors

    def test_contains_all_segments(self, catalog):
        all_segs = {e.segment.value for e in catalog.entries}
        assert "consumer" in all_segs
        assert "data_center" in all_segs
        assert "workstation" in all_segs

    def test_total_entries(self, catalog):
        assert len(catalog.entries) >= 40

    def test_lookup_exact_full_name(self, catalog):
        entry = catalog.lookup("NVIDIA GeForce RTX 4090")
        assert entry is not None
        assert entry.model_short == "RTX 4090"
        assert entry.vram_gib == 24.0

    def test_lookup_short_name(self, catalog):
        entry = catalog.lookup("RTX 4090")
        assert entry is not None
        assert entry.vram_gib == 24.0

    def test_lookup_alias(self, catalog):
        entry = catalog.lookup("4090")
        assert entry is not None
        assert "RTX 4090" in entry.model_full

    def test_lookup_h100(self, catalog):
        entry = catalog.lookup("NVIDIA H100-SXM-80GB")
        assert entry is not None
        assert entry.segment.value == "data_center"
        assert entry.vram_gib == 80.0

    def test_lookup_fuzzy_match(self, catalog):
        entry = catalog.lookup("h100")
        assert entry is not None
        assert entry.model_short == "H100"

    def test_lookup_unknown_returns_none(self, catalog):
        entry = catalog.lookup("Fake GPU Model X9000")
        assert entry is None

    def test_query_by_vendor(self, catalog):
        results = catalog.query(vendor="nvidia")
        all_nvidia = all(r["vendor"] == "nvidia" for r in results)
        assert all_nvidia
        assert len(results) > 20

    def test_query_by_segment(self, catalog):
        results = catalog.query(segment="data_center")
        all_dc = all(r["segment"] == "data_center" for r in results)
        assert all_dc

    def test_query_min_vram(self, catalog):
        results = catalog.query(min_vram=48)
        all_big = all(r["vram_gib"] >= 48 for r in results)
        assert all_big
        assert len(results) >= 5

    def test_query_capabilities_av1(self, catalog):
        results = catalog.query(capabilities=["av1_encode"])
        all_av1 = all("av1_encode" in r["capabilities"] for r in results)
        assert all_av1

    def test_query_capabilities_ray_tracing(self, catalog):
        results = catalog.query(capabilities=["ray_tracing"])
        all_rt = all("ray_tracing" in r["capabilities"] for r in results)
        assert all_rt

    def test_filter_by_capability(self, catalog):
        av1_gpus = catalog.filter_by_capability("av1_encode")
        assert len(av1_gpus) > 10

    def test_get_training_capable(self, catalog):
        capable = catalog.get_training_capable()
        assert len(capable) > 5
        for g in capable:
            assert g.tensor_tflops_fp16 >= 50
            assert g.vram_gib >= 16

    def test_get_inference_capable(self, catalog):
        capable = catalog.get_inference_capable()
        assert len(capable) > 10
        for g in capable:
            assert g.shader_tflops_fp32 >= 10
            assert g.vram_gib >= 8

    def test_group_by_vendor(self, catalog):
        groups = catalog.group_by_vendor()
        assert "nvidia" in groups
        assert "amd" in groups
        assert "intel" in groups

    def test_group_by_segment(self, catalog):
        groups = catalog.group_by_segment()
        assert "consumer" in groups
        assert "data_center" in groups
        assert "workstation" in groups

    def test_lookup_function(self):
        entry = lookup_gpu("RTX 5090")
        assert entry is not None
        assert entry.vram_gib == 32.0

    def test_lookup_amd_rx_7900xtx(self):
        entry = lookup_gpu("RX 7900 XTX")
        assert entry is not None
        assert entry.vendor.value == "amd"

    def test_lookup_intel_arc(self):
        entry = lookup_gpu("Arc A770")
        assert entry is not None
        assert entry.vendor.value == "intel"

    def test_datacenter_gpus_have_no_display(self):
        catalog = get_gpu_catalog()
        for e in catalog.entries:
            if e.segment.value == "data_center":
                assert e.capabilities.display_outputs is False

    def test_ecc_gpus_mark_ecc(self):
        catalog = get_gpu_catalog()
        ecc_gpus = [e for e in catalog.entries if e.capabilities.ecc_memory]
        assert len(ecc_gpus) > 5

    def test_to_dict(self, catalog):
        entry = catalog.lookup("RTX 4090")
        assert entry is not None
        d = entry.to_dict()
        assert d["model_short"] == "RTX 4090"
        assert d["vram_gib"] == 24.0
        assert "capabilities" in d
        assert isinstance(d["capabilities"], list)


class TestGpuCatalogEndpoints:
    def test_list_catalog_endpoint(self, client):
        resp = client.get("/api/v1/ml/gpu-catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 40

    def test_catalog_stats_endpoint(self, client):
        resp = client.get("/api/v1/ml/gpu-catalog/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entries"] >= 40
        assert "by_vendor" in data

    def test_lookup_endpoint(self, client):
        resp = client.get("/api/v1/ml/gpu-catalog/lookup", params={"name": "RTX 4090"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["vram_gib"] == 24.0

    def test_lookup_unknown_endpoint(self, client):
        resp = client.get("/api/v1/ml/gpu-catalog/lookup", params={"name": "Fake GPU"})
        assert resp.status_code == 200
        assert resp.json() is None

    def test_query_vendor_filter(self, client):
        resp = client.get("/api/v1/ml/gpu-catalog", params={"vendor": "amd"})
        data = resp.json()
        assert all(g["vendor"] == "amd" for g in data)

    def test_query_segment_filter(self, client):
        resp = client.get("/api/v1/ml/gpu-catalog", params={"segment": "data_center"})
        data = resp.json()
        assert all(g["segment"] == "data_center" for g in data)

    def test_query_capabilities_filter(self, client):
        resp = client.get("/api/v1/ml/gpu-catalog", params={"capabilities": "av1_encode"})
        data = resp.json()
        assert all("av1_encode" in g["capabilities"] for g in data)

    def test_schedule_with_capability_endpoint(self, client):
        resp = client.post("/api/v1/ml/schedule-with-capability", params={
            "name": "test-ai-job",
            "required_gpus": 2,
            "required_memory_gib": 32.0,
            "required_capabilities": "tensor_cores,av1_encode",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "capability_check" in data
