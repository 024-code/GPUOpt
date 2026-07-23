from __future__ import annotations

import pytest

from gpuopt.cpu_catalog import CpuCatalog, get_cpu_catalog, lookup_cpu, CpuVendor


@pytest.fixture
def catalog():
    return get_cpu_catalog()


class TestCpuCatalog:
    def test_contains_all_vendors(self, catalog):
        vendors = {e.vendor.value for e in catalog.entries}
        assert "intel" in vendors
        assert "amd" in vendors

    def test_total_entries(self, catalog):
        assert len(catalog.entries) >= 50

    def test_lookup_exact_full_name_intel(self, catalog):
        entry = catalog.lookup("Intel Core i9-14900K")
        assert entry is not None
        assert entry.model_short == "i9-14900K"
        assert entry.cores == 24
        assert entry.threads == 32

    def test_lookup_exact_full_name_amd(self, catalog):
        entry = catalog.lookup("AMD Ryzen 7 5700G")
        assert entry is not None
        assert "5700g" in entry.model_short.lower()
        assert entry.igpu is not None

    def test_lookup_short_name(self, catalog):
        entry = catalog.lookup("i9-13900K")
        assert entry is not None
        assert entry.boost_clock_ghz == 5.8

    def test_lookup_alias(self, catalog):
        entry = catalog.lookup("14900k")
        assert entry is not None
        assert "14900K" in entry.model_full

    def test_lookup_unknown_returns_none(self, catalog):
        entry = catalog.lookup("Fake CPU Model X9000")
        assert entry is None

    def test_lookup_by_igpu_pci_uhd770(self, catalog):
        entry = catalog.lookup_by_igpu_pci("0x4680")
        assert entry is not None
        assert entry.igpu is not None
        assert entry.igpu.execution_units == 32

    def test_lookup_by_igpu_pci_radeon780m(self, catalog):
        entry = catalog.lookup_by_igpu_pci("0x15BF")
        assert entry is not None
        assert entry.igpu is not None
        assert entry.igpu.compute_units == 12

    def test_lookup_by_igpu_pci_unknown(self, catalog):
        entry = catalog.lookup_by_igpu_pci("0xFFFF")
        assert entry is None

    def test_has_igpu_filter(self, catalog):
        results = catalog.query(has_igpu=True)
        assert len(results) > 0
        for r in results:
            assert r.get("igpu") is not None

    def test_min_cores_filter(self, catalog):
        results = catalog.query(min_cores=16)
        for r in results:
            assert r["cores"] >= 16

    def test_vendor_filter(self, catalog):
        results = catalog.query(vendor="amd")
        for r in results:
            assert r["vendor"] == "amd"

    def test_socket_filter(self, catalog):
        results = catalog.query(socket="am5")
        for r in results:
            assert r["socket"] == "am5"

    def test_intel_cpu_has_igpu(self, catalog):
        entry = catalog.lookup("i9-14900K")
        assert entry is not None
        assert entry.igpu is not None
        assert entry.igpu.execution_units == 32
        assert entry.igpu.pci_device_id == "0xA780"

    def test_amd_igpu_spec(self, catalog):
        entry = catalog.lookup("Ryzen 7 8700G")
        assert entry is not None
        igpu = entry.igpu
        assert igpu is not None
        assert igpu.compute_units == 12
        assert igpu.shader_tflops_fp32 == 8.9
        assert igpu.supports_av1_decode is True
        assert igpu.supports_av1_encode is True

    def test_xeon_ecc_support(self, catalog):
        entry = catalog.lookup("E-2388G")
        assert entry is not None
        assert entry.ecc_support is True

    def test_epyc_ecc_support(self, catalog):
        entry = catalog.lookup("EPYC 4564P")
        assert entry is not None
        assert entry.ecc_support is True

    def test_media_capabilities(self, catalog):
        entry = catalog.lookup("i5-14600K")
        assert entry is not None
        igpu = entry.igpu
        assert igpu is not None
        caps = igpu.media_capabilities()
        assert "av1_decode" in caps
        assert "hevc_decode" in caps
        assert "hevc_encode" in caps

    def test_11th_gen_no_av1(self, catalog):
        entry = catalog.lookup("i9-11900K")
        assert entry is not None
        igpu = entry.igpu
        assert igpu is not None
        assert igpu.supports_av1_decode is False

    def test_group_by_vendor(self, catalog):
        groups = catalog.group_by_vendor()
        assert "intel" in groups
        assert "amd" in groups
        assert len(groups["intel"]) > 0
        assert len(groups["amd"]) > 0

    def test_group_by_socket(self, catalog):
        groups = catalog.group_by_socket()
        assert "lga1700" in groups
        assert "am5" in groups

    def test_to_dict_structure(self, catalog):
        entry = catalog.lookup("i9-14900K")
        assert entry is not None
        d = entry.to_dict()
        assert d["vendor"] == "intel"
        assert d["cores"] == 24
        assert d["threads"] == 32
        assert d["base_clock_ghz"] == 3.2
        assert d["boost_clock_ghz"] == 6.0
        assert d["tdp_watts"] == 125
        igpu = d.get("igpu")
        assert igpu is not None
        assert igpu["execution_units"] == 32

    def test_lookup_function(self):
        entry = lookup_cpu("i7-14700K")
        assert entry is not None
        assert entry.cores == 20

    def test_server_xeon_scalable(self, catalog):
        entry = catalog.lookup("Platinum 8490H")
        assert entry is not None
        assert entry.cores == 60
        assert entry.threads == 120
        assert entry.ecc_support is True
        assert entry.igpu is None

    def test_server_epyc_genoa(self, catalog):
        entry = catalog.lookup("EPYC 9654")
        assert entry is not None
        assert entry.cores == 96
        assert entry.ecc_support is True
        assert entry.igpu is None

    def test_server_epyc_turin(self, catalog):
        entry = catalog.lookup("EPYC 9965")
        assert entry is not None
        assert entry.cores == 192
        assert entry.threads == 384

    def test_threadripper(self, catalog):
        entry = catalog.lookup("TR 7980X")
        assert entry is not None
        assert entry.cores == 64
        assert entry.threads == 128
        assert entry.igpu is None

    def test_core_ultra_arrow_lake(self, catalog):
        entry = catalog.lookup("Ultra 9 285K")
        assert entry is not None
        assert entry.hybrid_cores_p == 8
        assert entry.hybrid_cores_e == 16
        assert entry.cores == 24
        assert entry.threads == 24
        assert entry.igpu is not None
        assert entry.igpu.execution_units == 32

    def test_core_ultra_meteor_lake(self, catalog):
        entry = catalog.lookup("Ultra 9 185H")
        assert entry is not None
        assert entry.hybrid_cores_p == 6
        assert entry.hybrid_cores_e == 8
        assert entry.cores == 16
        assert entry.threads == 22

    def test_ryzen_ai_strix_point(self, catalog):
        entry = catalog.lookup("Ryzen AI 9 HX 370")
        assert entry is not None
        assert entry.hybrid_cores_p == 4
        assert entry.hybrid_cores_e == 8
        assert entry.igpu is not None
        assert entry.igpu.compute_units == 16
        assert entry.igpu.shader_tflops_fp32 == 11.9

    def test_pl1_pl2_fields(self, catalog):
        entry = catalog.lookup("Ultra 9 285K")
        assert entry is not None
        assert entry.pl1_watts == 125.0
        assert entry.pl2_watts == 250.0

    def test_hybrid_fields_in_to_dict(self, catalog):
        entry = catalog.lookup("Ultra 9 285K")
        assert entry is not None
        d = entry.to_dict()
        assert d["hybrid_cores_p"] == 8
        assert d["hybrid_cores_e"] == 16
        assert d["pl1_watts"] == 125.0
        assert d["pl2_watts"] == 250.0
