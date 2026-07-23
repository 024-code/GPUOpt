from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gpuopt import memory_catalog
from gpuopt.agent import detect_local_hardware, run_agent_once
from gpuopt.discovery_service import (
    AutoDiscoveryService,
    DiscoveryResult,
    NodeDiscoveryTarget,
    discover_cluster,
    discover_node,
    discover_via_http,
    discover_via_ssh,
)
from gpuopt.memory_catalog import (
    extract_spd_cas_latency_byte,
    extract_spd_cas_latency_from_decode_dimms,
    extract_spd_cas_latency_from_raw,
    parse_spd_hex_dump,
)


class TestSpdParsing:
    def test_parse_i2cdump_format(self):
        dump = (
            "0000: 23 10 0c 01 21 08 0c 00 00 00 05 0a 0c 00 00 00\n"
            "0010: 00 00 28 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        )
        result = parse_spd_hex_dump(dump)
        assert result[0x00] == 0x23
        assert result[0x10] == 0x00
        assert result[0x12] == 0x28

    def test_parse_xxd_format(self):
        dump = (
            "00000000: 2310 0c01 2108 0c00 0000 050a 0c00 0000  #...!...........\n"
            "00000010: 0000 2800 0000 0000 0000 0000 0000 0000  ..(.............\n"
        )
        result = parse_spd_hex_dump(dump)
        assert result[0x00] == 0x23
        assert result[0x12] == 0x28

    def test_parse_empty(self):
        assert parse_spd_hex_dump("") == {}

    def test_parse_garbage(self):
        assert parse_spd_hex_dump("not a hex dump") == {}

    def test_extract_spd_cas_latency_byte_ddr4(self):
        dump = "0010: 00 00 28 00 00 00 00 00\n"
        val = extract_spd_cas_latency_byte(dump, 18)
        assert val == 0x28

    def test_extract_spd_cas_latency_byte_missing(self):
        dump = "0000: 00 01 02\n"
        val = extract_spd_cas_latency_byte(dump, 99)
        assert val is None

    def test_extract_from_raw_ddr4(self):
        raw = bytes([0] * 18 + [28] + [0] * 10)
        result = extract_spd_cas_latency_from_raw(raw, speed_mts=0)
        assert result["dram_type"] == "DDR4"
        assert result["cas_latency_cycles"] == 28
        assert result["byte_18_hex"].lower() == "0x1c"

    def test_extract_from_raw_ddr5_no_speed(self):
        raw = bytes([0] * 0x0A + [0x05] + [0] * 7 + [0x28, 0x00] + [0] * 20)
        result = extract_spd_cas_latency_from_raw(raw, speed_mts=0)
        assert result["dram_type"] == "DDR5"
        assert "cas_latency_cycles" not in result

    def test_extract_from_raw_ddr5_with_speed(self):
        # DDR5-4800 CL40: tCKAVGmin MTB value ≈ 133 (0x85)
        raw = bytes([0] * 0x0A + [0x05] + [0] * 7 + [0x85, 0x00] + [0] * 20)
        result = extract_spd_cas_latency_from_raw(raw, speed_mts=4800)
        assert result["dram_type"] == "DDR5"
        assert result.get("cas_latency_cycles", 0) > 0

    def test_extract_from_raw_short(self):
        raw = bytes([0] * 10)
        result = extract_spd_cas_latency_from_raw(raw)
        assert "error" in result

    def test_extract_from_decode_dimms(self):
        text = (
            "Memory Type: DDR5\n"
            "Speed: 4800 MHz\n"
            "CAS Latency (CL): 40\n"
            "tRCD: 40\n"
            "tRP: 40\n"
            "tRAS: 80\n"
        )
        result = extract_spd_cas_latency_from_decode_dimms(text)
        assert result["cas_latency"] == 40
        assert result["tRCD"] == 40
        assert result["tRP"] == 40
        assert result["tRAS"] == 80
        assert result["speed"] == 4800
        assert result["memory_type"] == "DDR5"

    def test_extract_from_decode_dimms_empty(self):
        assert extract_spd_cas_latency_from_decode_dimms("") == {}


class TestAgent:
    def test_detect_local_hardware(self):
        hw = detect_local_hardware()
        assert "hostname" in hw
        assert "cpu" in hw

    def test_run_agent_once(self):
        hw = detect_local_hardware()
        assert isinstance(hw, dict)
        assert len(hw) > 0

    def test_run_agent_once_output(self):
        hw = detect_local_hardware()
        assert "hostname" in hw
        assert isinstance(hw, dict)


class TestDiscoveryService:
    def test_node_discovery_target_defaults(self):
        t = NodeDiscoveryTarget(host="10.0.0.1")
        assert t.host == "10.0.0.1"
        assert t.node_id == "10.0.0.1"
        assert t.ssh_key_file is None
        assert t.agent_endpoint is None

    def test_node_discovery_target_with_id(self):
        t = NodeDiscoveryTarget(host="10.0.0.2", node_id="gpu-node-2")
        assert t.node_id == "gpu-node-2"

    def test_discovery_result_success(self):
        r = DiscoveryResult("node-1", "10.0.0.1", True, hardware={"cpu": 4})
        assert r.success is True
        assert r.hardware["cpu"] == 4
        d = r.to_dict()
        assert d["node_id"] == "node-1"

    def test_discovery_result_failure(self):
        r = DiscoveryResult("node-2", "10.0.0.2", False, error="timeout")
        assert r.success is False
        assert "timeout" in r.error

    def test_discovery_via_ssh_no_output(self):
        t = NodeDiscoveryTarget(host="10.0.0.99")
        with patch("gpuopt.discovery_service._run_remote", return_value=""):
            r = discover_via_ssh(t)
            assert r.success is False
            assert "no output" in r.error

    def test_discovery_via_ssh_json_output(self):
        t = NodeDiscoveryTarget(host="10.0.0.1")
        with patch("gpuopt.discovery_service._run_remote", return_value='{"hostname": "node1"}'):
            r = discover_via_ssh(t)
            assert r.success is True
            assert r.hardware["hostname"] == "node1"

    def test_discovery_via_ssh_bad_json(self):
        t = NodeDiscoveryTarget(host="10.0.0.1")
        with patch("gpuopt.discovery_service._run_remote", return_value="not json"):
            r = discover_via_ssh(t)
            assert r.success is False

    def _mock_http_response(self, status: int, body: bytes = b""):
        m = MagicMock()
        cm = MagicMock()
        cm.status = status
        cm.read.return_value = body
        m.return_value.__enter__.return_value = cm
        return m

    def test_discovery_via_http_success(self):
        t = NodeDiscoveryTarget(host="10.0.0.1", agent_endpoint="http://10.0.0.1:8000")
        with patch("urllib.request.urlopen", self._mock_http_response(200, b'{"hostname": "node1"}')):
            r = discover_via_http(t)
            assert r.success is True

    def test_discovery_via_http_failure(self):
        t = NodeDiscoveryTarget(host="10.0.0.1", agent_endpoint="http://10.0.0.1:8000")
        with patch("urllib.request.urlopen", self._mock_http_response(500)):
            r = discover_via_http(t)
            assert r.success is False

    def test_discover_node_uses_http_when_endpoint_set(self):
        t = NodeDiscoveryTarget(host="10.0.0.1", agent_endpoint="http://10.0.0.1:8000")
        with patch("gpuopt.discovery_service.discover_via_http") as mock_http:
            mock_http.return_value = DiscoveryResult("n1", "10.0.0.1", True)
            r = discover_node(t)
            mock_http.assert_called_once()
            assert r.success is True

    def test_discover_node_uses_ssh_when_no_endpoint(self):
        t = NodeDiscoveryTarget(host="10.0.0.1")
        with patch("gpuopt.discovery_service.discover_via_ssh") as mock_ssh:
            mock_ssh.return_value = DiscoveryResult("n1", "10.0.0.1", True)
            r = discover_node(t)
            mock_ssh.assert_called_once()
            assert r.success is True

    def test_discover_cluster(self):
        targets = [
            NodeDiscoveryTarget(host="10.0.0.1"),
            NodeDiscoveryTarget(host="10.0.0.2"),
        ]
        with patch("gpuopt.discovery_service.discover_node") as mock_d:
            mock_d.return_value = DiscoveryResult("n1", "10.0.0.1", True)
            results = discover_cluster(targets, concurrency=2)
            assert len(results) == 2

    def test_auto_discovery_add_remove(self):
        svc = AutoDiscoveryService()
        assert len(svc.targets) == 0
        svc.add_target(NodeDiscoveryTarget(host="10.0.0.1"))
        assert len(svc.targets) == 1
        svc.remove_target("10.0.0.1")
        assert len(svc.targets) == 0

    @pytest.mark.asyncio
    async def test_auto_discovery_start_stop(self):
        svc = AutoDiscoveryService(
            targets=[NodeDiscoveryTarget(host="10.0.0.1")],
            interval_seconds=1,
        )
        assert svc._task is None
        await svc.start()
        assert svc._task is not None
        await svc.stop()
        assert svc._task is None

    @pytest.mark.asyncio
    async def test_auto_discovery_discover_async(self):
        svc = AutoDiscoveryService(
            targets=[NodeDiscoveryTarget(host="10.0.0.1")],
        )
        with patch("gpuopt.discovery_service.discover_node") as mock_d:
            mock_d.return_value = DiscoveryResult("n1", "10.0.0.1", True)
            results = await svc.discover_async()
            assert len(results) == 1
            assert results[0].success is True
