from __future__ import annotations

import pytest
from unittest.mock import patch

from gpuopt.hardware_detector import (
    detect_local_cpu,
    detect_local_memory,
    _extract_dmi_field,
    _extract_dmi_int,
    _parse_dmi_size,
)


class TestParseDmiSize:
    def test_mb(self):
        assert _parse_dmi_size("8192 MB") == 8192

    def test_gb(self):
        assert _parse_dmi_size("16 GB") == 16384

    def test_none(self):
        assert _parse_dmi_size("No Module Installed") == 0

    def test_empty(self):
        assert _parse_dmi_size("") == 0


class TestExtractDmiField:
    def test_basic_field(self):
        section = (
            "Memory Device\n"
            "\tSize: 16384 MB\n"
            "\tType: DDR5\n"
            "\tSpeed: 5600 MHz\n"
        )
        assert _extract_dmi_field(section, "Size") == "16384 MB"
        assert _extract_dmi_field(section, "Type") == "DDR5"
        assert _extract_dmi_field(section, "Speed") == "5600 MHz"

    def test_missing_field(self):
        section = "Memory Device\n\tSize: 16384 MB\n"
        assert _extract_dmi_field(section, "Locator") == ""


class TestExtractDmiInt:
    def test_mhz_field(self):
        section = "Memory Device\n\tSpeed: 5600 MHz\n"
        assert _extract_dmi_int(section, "Speed") == 5600

    def test_mb_field(self):
        section = "Memory Device\n\tSize: 16384 MB\n"
        assert _extract_dmi_int(section, "Size") == 16384

    def test_plain_int_field(self):
        section = "Memory Device\n\tRank: 2\n"
        assert _extract_dmi_int(section, "Rank") == 2

    def test_missing_int(self):
        section = "Memory Device\n\tSpeed: Unknown\n"
        assert _extract_dmi_int(section, "Speed") == 0


class TestDetectLocalCpu:
    def test_with_lscpu_output(self):
        mock_lscpu = (
            "Architecture:            x86_64\n"
            "CPU(s):                  16\n"
            "Thread(s) per core:      2\n"
            "Model name:              Intel Core i9-14900K\n"
            "CPU max MHz:             6000.0000\n"
            "CPU min MHz:             800.0000\n"
        )
        with patch("gpuopt.hardware_detector._run", return_value=mock_lscpu):
            result = detect_local_cpu()
        assert result["architecture"] == "x86_64"
        assert result["cores"] == 8
        assert result["threads"] == 16
        assert "14900K" in result["model"]
        assert result["max_frequency_mhz"] == 6000

    def test_with_amd_lscpu(self):
        mock_lscpu = (
            "Architecture:            x86_64\n"
            "CPU(s):                  16\n"
            "Thread(s) per core:      2\n"
            "Model name:              AMD Ryzen 7 8700G\n"
            "CPU max MHz:             5100.0000\n"
            "CPU min MHz:             3000.0000\n"
        )
        with patch("gpuopt.hardware_detector._run", return_value=mock_lscpu):
            result = detect_local_cpu()
        assert "8700G" in result["model"]
        assert result["catalog_match"] is not None
        assert result["catalog_match"]["cores"] == 8
        assert result["igpu_model"] == "Radeon 780M"

    def test_fallback_platform_processor(self):
        with patch("gpuopt.hardware_detector._run", return_value=""):
            with patch("platform.processor", return_value="arm64"):
                result = detect_local_cpu()
        assert result["model"] == "arm64"

    def test_catalog_match_for_intel(self):
        mock_lscpu = (
            "Architecture:            x86_64\n"
            "CPU(s):                  16\n"
            "Thread(s) per core:      2\n"
            "Model name:              Intel Core i5-14600K\n"
        )
        with patch("gpuopt.hardware_detector._run", return_value=mock_lscpu):
            result = detect_local_cpu()
        assert result["catalog_match"] is not None
        assert result["catalog_match"]["cores"] == 14
        assert result["igpu_model"] == "UHD 770"

    def test_catalog_match_for_amd_igpu(self):
        mock_lscpu = (
            "Architecture:            x86_64\n"
            "CPU(s):                  8\n"
            "Thread(s) per core:      2\n"
            "Model name:              AMD Ryzen 5 8600G\n"
        )
        with patch("gpuopt.hardware_detector._run", return_value=mock_lscpu):
            result = detect_local_cpu()
        assert result["catalog_match"] is not None
        assert result["catalog_match"]["cores"] == 6
        assert result["igpu_model"] == "Radeon 760M"

    def test_no_catalog_match(self):
        mock_lscpu = (
            "Architecture:            x86_64\n"
            "CPU(s):                  4\n"
            "Thread(s) per core:      1\n"
            "Model name:              Generic Unrecognized CPU\n"
        )
        with patch("gpuopt.hardware_detector._run", return_value=mock_lscpu):
            result = detect_local_cpu()
        assert result["catalog_match"] is None
        assert result["igpu_model"] == ""


class TestDetectLocalMemory:
    def test_dmidecode_ddr5(self):
        mock_dmi = (
            "Memory Device\n"
            "\tLocator: DIMM_A1\n"
            "\tBank Locator: BANK 0\n"
            "\tManufacturer: Samsung\n"
            "\tSerial Number: 12345678\n"
            "\tSize: 16384 MB\n"
            "\tType: DDR5\n"
            "\tSpeed: 5600 MHz\n"
            "\tConfigured Memory Speed: 5600 MHz\n"
            "\tForm Factor: 9\n"
            "\tData Width: 64 bits\n"
            "\tTotal Width: 72 bits\n"
            "\tRank: 2\n"
            "\tConfigured Voltage: 1.1 V\n"
            "\n\n"
            "Memory Device\n"
            "\tLocator: DIMM_B1\n"
            "\tBank Locator: BANK 1\n"
            "\tManufacturer: Samsung\n"
            "\tSize: 16384 MB\n"
            "\tType: DDR5\n"
            "\tSpeed: 5600 MHz\n"
            "\tConfigured Memory Speed: 5600 MHz\n"
            "\tForm Factor: 9\n"
            "\tData Width: 64 bits\n"
            "\tTotal Width: 72 bits\n"
            "\tRank: 2\n"
        )
        with patch("gpuopt.hardware_detector._run", return_value=mock_dmi):
            mem = detect_local_memory()
        assert mem.total_size_gb == 32.0
        assert mem.memory_type.value == "DDR5"
        assert mem.ecc_enabled is True
        assert mem.num_dimms == 2
        assert mem.max_speed_mhz == 5600

    def test_empty_dmidecode_fallback(self):
        with patch("gpuopt.hardware_detector._run", return_value=""):
            mem = detect_local_memory()
        assert mem.total_size_gb > 0
        assert mem.memory_type.value == "Unknown"

    def test_no_dimms_populated(self):
        mock_dmi = "Memory Device\n\tSize: No Module Installed\n\n"
        with patch("gpuopt.hardware_detector._run", return_value=mock_dmi):
            mem = detect_local_memory()
        assert mem.num_dimms == 0


class TestDetectLocalGpuPciIds:
    def test_with_lspci(self):
        mock_lspci = (
            "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA102 [GeForce RTX 3080] [10de:2206] (rev a1)\n"
            "02:00.0 Display controller [0380]: Intel Corporation Raptor Lake-S UHD Graphics [8086:a780] (rev 04)\n"
        )
        with patch("gpuopt.hardware_detector._run", return_value=mock_lspci):
            from gpuopt.hardware_detector import detect_local_gpu_pci_ids
            ids = detect_local_gpu_pci_ids()
        assert "10de:2206" in ids
        assert "8086:a780" in ids

    def test_empty_lspci(self):
        with patch("gpuopt.hardware_detector._run", return_value=""):
            from gpuopt.hardware_detector import detect_local_gpu_pci_ids
            ids = detect_local_gpu_pci_ids()
        assert ids == []
