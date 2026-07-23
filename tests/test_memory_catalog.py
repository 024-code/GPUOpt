from __future__ import annotations

import pytest

from gpuopt.memory_catalog import (
    MemoryType,
    FormFactor,
    EccStatus,
    MemoryCardinality,
    SmbiosDimmInfo,
    SystemMemorySummary,
    classify_speed,
    memory_type_from_speed,
    ddr_peak_bandwidth_gbps,
    decode_manufacturer,
    decode_part_number,
    detect_ecc,
    detect_registered,
)


class TestMemoryType:
    def test_from_smbios_type_ddr4(self):
        assert MemoryType.from_smbios_type(0x1A) == MemoryType.DDR4

    def test_from_smbios_type_ddr5(self):
        assert MemoryType.from_smbios_type(0x22) == MemoryType.DDR5

    def test_from_smbios_type_unknown(self):
        assert MemoryType.from_smbios_type(0xFF) == MemoryType.UNKNOWN

    def test_from_name_ddr4(self):
        assert MemoryType.from_name("DDR4") == MemoryType.DDR4

    def test_from_name_ddr5(self):
        assert MemoryType.from_name("DDR5") == MemoryType.DDR5

    def test_from_name_lpddr5(self):
        assert MemoryType.from_name("LPDDR5") == MemoryType.LPDDR5

    def test_from_name_case_insensitive(self):
        assert MemoryType.from_name("ddr4") == MemoryType.DDR4


class TestFormFactor:
    def test_from_code_dimm(self):
        assert FormFactor.from_smbios_code(9) == FormFactor.DIMM

    def test_from_code_sodimm(self):
        assert FormFactor.from_smbios_code(12) == FormFactor.SODIMM

    def test_from_code_rdimm(self):
        assert FormFactor.from_smbios_code(15) == FormFactor.RDIMM

    def test_from_code_camm2(self):
        assert FormFactor.from_smbios_code(27) == FormFactor.CAMM2

    def test_from_code_unknown(self):
        assert FormFactor.from_smbios_code(99) == FormFactor.UNKNOWN


class TestClassifySpeed:
    def test_ddr4_3200(self):
        assert classify_speed(3200, "DDR4") == "DDR4-3200"

    def test_ddr5_5600(self):
        assert classify_speed(5600, "DDR5") == "DDR5-5600"

    def test_ddr5_4800(self):
        assert classify_speed(4800, "DDR5") == "DDR5-4800"

    def test_auto_detect_ddr4_2666(self):
        classified = classify_speed(2666)
        assert "DDR4" in classified

    def test_auto_detect_ddr5_6000(self):
        classified = classify_speed(6000)
        assert classified == "DDR5-6000"

    def test_nearest_bin(self):
        assert classify_speed(3233) in ("DDR4-3200", "DDR5-3200")


class TestMemoryTypeFromSpeed:
    def test_2133_is_ddr4(self):
        assert memory_type_from_speed(2133) == MemoryType.DDR4

    def test_3200_is_ddr4(self):
        assert memory_type_from_speed(3200) == MemoryType.DDR4

    def test_4800_is_ddr5(self):
        assert memory_type_from_speed(4800) == MemoryType.DDR5

    def test_6000_is_ddr5(self):
        assert memory_type_from_speed(6000) == MemoryType.DDR5


class TestBandwidth:
    def test_ddr5_4800_64bit(self):
        bw = ddr_peak_bandwidth_gbps(4800, 64)
        assert bw == 38.4

    def test_ddr4_3200_64bit(self):
        bw = ddr_peak_bandwidth_gbps(3200, 64)
        assert bw == 25.6

    def test_zero_bus(self):
        assert ddr_peak_bandwidth_gbps(4800, 0) == 0.0


class TestEccDetection:
    def test_no_ecc(self):
        assert detect_ecc(64, 64) == EccStatus.NONE

    def test_ecc_enabled(self):
        assert detect_ecc(72, 64) == EccStatus.ECC

    def test_unknown(self):
        assert detect_ecc(0, 0) == EccStatus.UNKNOWN


class TestRegisteredDetection:
    def test_rdimm(self):
        assert detect_registered(FormFactor.RDIMM) == "registered"

    def test_lrdimm(self):
        assert detect_registered(FormFactor.LRDIMM) == "registered"

    def test_udimm(self):
        assert detect_registered(FormFactor.UDIMM) == "unbuffered"

    def test_sodimm_default_unbuffered(self):
        assert detect_registered(FormFactor.SODIMM) == "unbuffered"

    def test_none(self):
        assert detect_registered(None) is None


class TestManufacturerDecode:
    def test_samsung_code(self):
        assert decode_manufacturer("80CE") == "Samsung"

    def test_hynix_code(self):
        assert decode_manufacturer("80AD") == "Hynix"

    def test_micron_code(self):
        assert decode_manufacturer("2C00") == "Micron"

    def test_kingston_part_number(self):
        assert decode_manufacturer(None, "KVR32N22S8/8") == "Kingston"

    def test_corsair_part_number(self):
        assert decode_manufacturer(None, "CMK32GX4M2B3200C16") == "Corsair"

    def test_gskill_part_number(self):
        assert decode_manufacturer(None, "F4-3200C16D-32GVK") == "G.Skill"

    def test_unknown_returns_unknown(self):
        result = decode_manufacturer(None, "SomeRandomPart123")
        assert result == "Unknown"


class TestPartNumberDecode:
    def test_samsung_ddr4(self):
        info = decode_part_number("M391A2K43BB1-CRC")
        assert info["vendor"] == "Samsung"

    def test_micron_ddr5(self):
        info = decode_part_number("MT60B2G8HB-48B")
        assert info["vendor"] == "Micron"
        assert "DDR5" in info.get("type", "")

    def test_hynix_ddr4(self):
        info = decode_part_number("HMA82GU6AFR8N-UH")
        assert info["vendor"] == "Hynix"

    def test_unknown(self):
        info = decode_part_number("XYZ123")
        assert info["vendor"] == "Unknown"


class TestSmbiosDimmInfo:
    def test_create_and_classify(self):
        dimm = SmbiosDimmInfo(
            locator="DIMM_A1",
            size_mb=16384,
            speed_mhz=5600,
            memory_type_code=34,
            form_factor_code=9,
            data_width_bits=64,
            total_width_bits=72,
            rank=2,
            part_number="M425R2G3BB0-CQK",
        )
        assert dimm.classify_type() == MemoryType.DDR5
        assert dimm.classify_form_factor() == FormFactor.DIMM
        assert dimm.ecc_status() == EccStatus.ECC
        assert dimm.is_ecc() is True
        assert dimm.cardinality() == MemoryCardinality.DUAL_RANK
        assert dimm.speed_label() == "DDR5-5600"
        assert dimm.bandwidth_gbps() == 44.8
        assert dimm.size_mb == 16384

    def test_non_ecc_dimm(self):
        dimm = SmbiosDimmInfo(
            locator="DIMM_B1",
            size_mb=8192,
            speed_mhz=3200,
            memory_type_code=26,
            form_factor_code=12,
            data_width_bits=64,
            total_width_bits=64,
            rank=1,
        )
        assert not dimm.is_ecc()
        assert dimm.cardinality() == MemoryCardinality.SINGLE_RANK
        assert dimm.speed_label() == "DDR4-3200"

    def test_to_dict(self):
        dimm = SmbiosDimmInfo(
            locator="DIMM_A1",
            size_mb=16384,
            speed_mhz=5600,
            memory_type_code=34,
            form_factor_code=9,
            data_width_bits=64,
            total_width_bits=72,
            rank=2,
        )
        d = dimm.to_dict()
        assert d["size_gb"] == 16.0
        assert d["ecc"] is True
        assert d["rank"] == 2
        assert d["speed_label"] == "DDR5-5600"
        assert d["memory_type"] == "DDR5"
        assert d["form_factor"] == "DIMM"


class TestSystemMemorySummary:
    def test_from_dimm_list(self):
        dimms = [
            SmbiosDimmInfo(locator="A1", size_mb=16384, speed_mhz=5600,
                           memory_type_code=34, form_factor_code=9,
                           data_width_bits=64, total_width_bits=72, rank=2),
            SmbiosDimmInfo(locator="B1", size_mb=16384, speed_mhz=5600,
                           memory_type_code=34, form_factor_code=9,
                           data_width_bits=64, total_width_bits=72, rank=2),
        ]
        sm = SystemMemorySummary.from_dimm_list(dimms)
        assert sm.total_size_gb == 32.0
        assert sm.memory_type == MemoryType.DDR5
        assert sm.form_factor == FormFactor.DIMM
        assert sm.ecc_enabled is True
        assert sm.num_dimms == 2
        assert sm.max_speed_mhz == 5600
        assert sm.num_channels >= 1

    def test_empty_list(self):
        sm = SystemMemorySummary.from_dimm_list([])
        assert sm.total_size_gb == 0.0
        assert sm.memory_type == MemoryType.UNKNOWN
        assert sm.form_factor == FormFactor.UNKNOWN

    def test_mixed_speed_dimm(self):
        dimms = [
            SmbiosDimmInfo(locator="A1", size_mb=8192, speed_mhz=3200,
                           memory_type_code=26, form_factor_code=9,
                           data_width_bits=64, total_width_bits=64, rank=1),
            SmbiosDimmInfo(locator="B1", size_mb=8192, speed_mhz=3200,
                           memory_type_code=26, form_factor_code=9,
                           data_width_bits=64, total_width_bits=64, rank=1),
        ]
        sm = SystemMemorySummary.from_dimm_list(dimms)
        assert sm.total_size_gb == 16.0
        assert sm.memory_type == MemoryType.DDR4
        assert sm.ecc_enabled is False

    def test_to_dict_structure(self):
        dimms = [
            SmbiosDimmInfo(locator="A1", size_mb=16384, speed_mhz=5600,
                           memory_type_code=34, form_factor_code=9,
                           data_width_bits=64, total_width_bits=72, rank=2),
        ]
        sm = SystemMemorySummary.from_dimm_list(dimms)
        d = sm.to_dict()
        assert d["total_size_gb"] == 16.0
        assert "dimms" in d
        assert len(d["dimms"]) == 1
        assert d["memory_type"] == "DDR5"
        assert d["ecc_enabled"] is True
