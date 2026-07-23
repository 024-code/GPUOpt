from __future__ import annotations

import logging
import os
import platform
import re
import subprocess as sp
from typing import Any

from .cpu_catalog import (
    CpuCatalog,
    CpuCatalogEntry,
    CpuSocket,
    CpuVendor,
    get_cpu_catalog,
)
from .memory_catalog import (
    FormFactor,
    MemoryCardinality,
    MemoryType,
    SmbiosDimmInfo,
    SystemMemorySummary,
    decode_manufacturer,
    decode_part_number,
    detect_ecc,
    detect_memory_from_sysfs,
)

logger = logging.getLogger(__name__)


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        r = sp.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (FileNotFoundError, sp.TimeoutExpired, PermissionError) as exc:
        logger.debug("Command %s failed: %s", cmd[0], exc)
        return ""


def _run_remote(host: str, cmd: list[str], timeout: int = 10, key_file: str | None = None) -> str:
    ssh_cmd = ["ssh"]
    if key_file:
        ssh_cmd += ["-i", key_file]
    ssh_cmd += ["-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", host, *cmd]
    try:
        r = sp.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (FileNotFoundError, sp.TimeoutExpired, PermissionError) as exc:
        logger.debug("Remote exec on %s failed: %s", host, exc)
        return ""


def _parse_lscpu_field(text: str, field: str) -> str:
    pat = re.compile(rf"^{re.escape(field)}:\s*(.*)", re.MULTILINE)
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def _parse_model_via_proc_cpuinfo(text: str = "") -> str:
    if not text:
        text = _run(["cat", "/proc/cpuinfo"])
    if not text:
        return ""
    for line in text.splitlines():
        if line.startswith("model name") or line.startswith("Model"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return ""


# ── CPU Detection ──

def detect_local_cpu() -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": "", "architecture": "", "cores": 0, "threads": 0,
        "base_frequency_mhz": 0, "max_frequency_mhz": 0,
        "socket": "", "vendor": "",
        "l1d_cache_kb": 0, "l1i_cache_kb": 0, "l2_cache_kb": 0, "l3_cache_kb": 0,
        "catalog_match": None, "igpu_pci_id": "", "igpu_model": "",
    }
    lscpu = _run(["lscpu"])
    if lscpu:
        result["architecture"] = _parse_lscpu_field(lscpu, "Architecture")
        cpus = int(_parse_lscpu_field(lscpu, "CPU(s)") or 0)
        threads_per_core = int(_parse_lscpu_field(lscpu, "Thread(s) per core") or 1)
        result["cores"] = cpus // max(threads_per_core, 1) if threads_per_core else cpus
        result["threads"] = cpus
        result["vendor"] = _parse_lscpu_field(lscpu, "Vendor ID")
        raw_model = _parse_lscpu_field(lscpu, "Model name")
        if raw_model:
            result["model"] = raw_model
        result["socket"] = _parse_lscpu_field(lscpu, "Socket(s)")
        model_line = _parse_lscpu_field(lscpu, "Model")
        if model_line:
            result["model_line"] = model_line
        max_mhz = _parse_lscpu_field(lscpu, "CPU max MHz")
        if max_mhz:
            result["max_frequency_mhz"] = round(float(max_mhz))
        min_mhz = _parse_lscpu_field(lscpu, "CPU min MHz")
        if min_mhz:
            result["base_frequency_mhz"] = round(float(min_mhz))
    else:
        result["model"] = platform.processor()
    if not result["model"]:
        result["model"] = _parse_model_via_proc_cpuinfo()
    if not result["model"]:
        result["model"] = platform.processor()

    for level, key in [("L1d", "l1d_cache_kb"), ("L1i", "l1i_cache_kb"),
                        ("L2", "l2_cache_kb"), ("L3", "l3_cache_kb")]:
        text = _run(["lscpu", "--cache"])
        if text:
            pat = re.compile(rf"{level}[^:]*:\s*(\d+)\s*K")
            m = pat.search(text)
            if m:
                result[key] = int(m.group(1))

    catalog = get_cpu_catalog()
    match = catalog.lookup(result["model"])
    if match:
        result["catalog_match"] = match.to_dict()
        result["igpu_model"] = match.igpu.model if match.igpu else ""
        result["igpu_pci_id"] = match.igpu.pci_device_id if match.igpu else ""
    return result


# ── NUMA Topology ──

def detect_numa_topology() -> dict[str, Any]:
    result: dict[str, Any] = {
        "numa_nodes": 0, "numa_memory": [], "cpu_to_numa": {},
        "has_numactl": False,
    }
    lscpu = _run(["lscpu", "-e"])
    if lscpu:
        lines = lscpu.strip().splitlines()
        if len(lines) >= 2:
            header = lines[0].lower().split()
            numa_idx = next((i for i, h in enumerate(header) if "numa" in h), -1)
            cpu_idx = next((i for i, h in enumerate(header) if "cpu" in h), -1)
            if numa_idx >= 0 and cpu_idx >= 0:
                nodes: set[str] = set()
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) > max(numa_idx, cpu_idx):
                        node = parts[numa_idx]
                        cpu = parts[cpu_idx]
                        nodes.add(node)
                        result["cpu_to_numa"][cpu] = node
                result["numa_nodes"] = len(nodes)
    if result["numa_nodes"] == 0:
        lscpu_flat = _run(["lscpu"])
        if lscpu_flat:
            numa_str = _parse_lscpu_field(lscpu_flat, "NUMA node(s)")
            if numa_str and numa_str.isdigit():
                result["numa_nodes"] = int(numa_str)

    numactl = _run(["numactl", "--hardware"])
    if numactl:
        result["has_numactl"] = True
        for line in numactl.splitlines():
            m = re.match(r"^node\s+(\d+)\s+(cpus|size):\s*(.*)", line, re.IGNORECASE)
            if m:
                node_id, kind, rest = m.group(1), m.group(2).lower(), m.group(3).strip()
                if kind == "size":
                    size_m = re.search(r"(\d+)\s*MB", rest, re.IGNORECASE)
                    if size_m:
                        result["numa_memory"].append({"node": int(node_id), "memory_mb": int(size_m.group(1))})
    return result


# ── GPU PCI & Name Detection ──

def detect_local_gpu_pci_ids() -> list[str]:
    lspci = _run(["lspci", "-nn"])
    ids: list[str] = []
    for line in lspci.splitlines():
        if any(kw in line.lower() for kw in ["vga compatible", "3d controller", "display controller"]):
            m = re.search(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]", line)
            if m:
                ids.append(m.group(1))
    return ids


def detect_gpu_names_nvidia_smi() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"], timeout=10)
    gpus: list[dict] = []
    if not out:
        return gpus
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "index": int(parts[0]) if parts[0].isdigit() else -1,
                "name": parts[1],
                "vram_mb": int(float(parts[2])) if parts[2].replace(".", "").isdigit() else 0,
                "util_pct": int(float(parts[3])) if parts[3].replace(".", "").isdigit() else 0,
                "temp_c": int(float(parts[4])) if parts[4].replace(".", "").isdigit() else 0,
                "source": "nvidia-smi",
            })
    return gpus


def detect_gpu_names_rocm_smi() -> list[dict]:
    out = _run(["rocm-smi", "--showallinfo"], timeout=10)
    gpus: list[dict] = []
    if not out:
        return gpus
    current: dict[str, Any] = {}
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d+)\]\s*:\s*(.*)", line)
        if m:
            if current:
                gpus.append(current)
            current = {"index": int(m.group(1)), "name": m.group(2).strip(), "source": "rocm-smi"}
        elif current:
            m2 = re.match(r"\s*VRAM\s*:\s*(\d+)\s*MB", line, re.IGNORECASE)
            if m2:
                current["vram_mb"] = int(m2.group(1))
            m3 = re.match(r"\s*Temperature\s*:\s*(\d+)\.?\d*\s*C", line, re.IGNORECASE)
            if m3:
                current["temp_c"] = int(m3.group(1))
    if current:
        gpus.append(current)
    return gpus


def detect_gpu_names_intel_xpu() -> list[dict]:
    out = _run(["xpu-smi", "discovery"], timeout=10)
    gpus: list[dict] = []
    if not out:
        return gpus
    for line in out.splitlines():
        m = re.search(r"Device\s+(\d+)\s*:\s*(.*)", line, re.IGNORECASE)
        if m:
            gpus.append({"index": int(m.group(1)), "name": m.group(2).strip(), "source": "xpu-smi"})
    return gpus


def detect_all_gpus() -> list[dict]:
    gpus = detect_gpu_names_nvidia_smi()
    if not gpus:
        gpus = detect_gpu_names_rocm_smi()
    if not gpus:
        gpus = detect_gpu_names_intel_xpu()
    if not gpus:
        pci_ids = detect_local_gpu_pci_ids()
        for i, pid in enumerate(pci_ids):
            from .gpu_catalog import lookup_gpu
            catalog_match = lookup_gpu(pid)
            gpus.append({
                "index": i,
                "pci_id": pid,
                "name": catalog_match.model_full if catalog_match else f"GPU ({pid})",
                "source": "pci_fallback",
            })
    return gpus


# ── SMBIOS Helpers ──

def _extract_dmi_field(section: str, field: str) -> str:
    pat = re.compile(rf"^\s*{re.escape(field)}:\s*(.*)", re.MULTILINE)
    m = pat.search(section)
    return m.group(1).strip() if m else ""


def _extract_dmi_int(section: str, field: str) -> int:
    val = _extract_dmi_field(section, field)
    val = val.replace("MB", "").replace("GB", "").replace("MHz", "").replace("bits", "").replace("ns", "").strip()
    try:
        return int(float(val))
    except ValueError:
        return 0


def _parse_dmi_size(size_str: str) -> int:
    size_str = size_str.strip()
    if not size_str or size_str in ("No Module Installed", "None"):
        return 0
    if "MB" in size_str:
        return int(float(size_str.replace("MB", "").strip()))
    if "GB" in size_str:
        return int(float(size_str.replace("GB", "").strip()) * 1024)
    try:
        return int(size_str)
    except ValueError:
        return 0


# ── Memory Detection ──

def detect_local_memory() -> SystemMemorySummary:
    dmidecode_out = _run(["dmidecode", "-t", "memory"], timeout=10)
    if not dmidecode_out:
        return _detect_memory_psutil_fallback()

    dimms: list[SmbiosDimmInfo] = []
    sections = re.split(r"\n\n+", dmidecode_out)

    for sec in sections:
        if not sec.startswith("Memory Device"):
            continue
        try:
            size_str = _extract_dmi_field(sec, "Size")
            size_mb = _parse_dmi_size(size_str)
            if size_mb == 0:
                continue
            dimm = SmbiosDimmInfo(
                locator=_extract_dmi_field(sec, "Locator"),
                bank_locator=_extract_dmi_field(sec, "Bank Locator"),
                manufacturer=_extract_dmi_field(sec, "Manufacturer"),
                part_number=_extract_dmi_field(sec, "Part Number"),
                serial_number=_extract_dmi_field(sec, "Serial Number"),
                size_mb=size_mb,
                speed_mhz=_extract_dmi_int(sec, "Speed"),
                configured_speed_mhz=_extract_dmi_int(sec, "Configured Memory Speed"),
                memory_type=_extract_dmi_field(sec, "Type"),
                memory_type_code=_extract_dmi_int(sec, "Type"),
                form_factor_code=_extract_dmi_int(sec, "Form Factor"),
                data_width_bits=_extract_dmi_int(sec, "Data Width"),
                total_width_bits=_extract_dmi_int(sec, "Total Width"),
                rank=_extract_dmi_int(sec, "Rank"),
                voltage=float(_extract_dmi_field(sec, "Configured Voltage").replace("V", "").strip())
                if _extract_dmi_field(sec, "Configured Voltage") else 0.0,
                manufacturer_id=_extract_dmi_field(sec, "Manufacturer") or "",
                cas_latency=_extract_dmi_field(sec, "Minimum CAS Latency"),
                tRCD=_extract_dmi_field(sec, "Minimum tRCD"),
                tRP=_extract_dmi_field(sec, "Minimum tRP"),
                tRAS=_extract_dmi_field(sec, "Minimum tRAS"),
                min_tcycle=_extract_dmi_field(sec, "Minimum Cycle Time"),
                pmic_manufacturer=_extract_dmi_field(sec, "PMIC Manufacturer") or _extract_dmi_field(sec, "Power Management IC Manufacturer"),
                pmic_part_number=_extract_dmi_field(sec, "PMIC Part Number") or _extract_dmi_field(sec, "Power Management IC Part Number"),
                thermal_sensor=bool(_extract_dmi_field(sec, "Thermal Sensor") or ""),
            )
            dimms.append(dimm)
        except Exception as exc:
            logger.debug("Failed to parse DMI memory section: %s", exc)
            continue

    if not dimms:
        return _detect_memory_psutil_fallback()
    return SystemMemorySummary.from_dimm_list(dimms)


def _detect_memory_psutil_fallback() -> SystemMemorySummary:
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = round(mem.total / (1024 ** 3), 1)
        return SystemMemorySummary(
            total_size_gb=total_gb,
            memory_type=MemoryType.UNKNOWN,
            form_factor=FormFactor.UNKNOWN,
            ecc_enabled=False,
            num_dimms=0,
        )
    except ImportError:
        pass
    return detect_memory_from_sysfs()


# ── BIOS & Motherboard Detection ──

def detect_bios_info() -> dict[str, str]:
    dmi = _run(["dmidecode", "-t", "bios"], timeout=5)
    if not dmi:
        return {}
    return {
        "vendor": _extract_dmi_field(dmi, "Vendor"),
        "version": _extract_dmi_field(dmi, "Version"),
        "release_date": _extract_dmi_field(dmi, "Release Date"),
        "uefi": "yes" if "UEFI" in dmi or "uefi" in dmi else "unknown",
    }


def detect_motherboard_info() -> dict[str, str]:
    dmi = _run(["dmidecode", "-t", "baseboard"], timeout=5)
    if not dmi:
        return {}
    return {
        "manufacturer": _extract_dmi_field(dmi, "Manufacturer"),
        "product_name": _extract_dmi_field(dmi, "Product Name"),
        "version": _extract_dmi_field(dmi, "Version"),
        "serial_number": _extract_dmi_field(dmi, "Serial Number"),
        "asset_tag": _extract_dmi_field(dmi, "Asset Tag"),
    }


# ── NIC & Storage Detection via lspci ──

def detect_pci_network_devices() -> list[dict]:
    lspci = _run(["lspci", "-nn"])
    devices: list[dict] = []
    for line in lspci.splitlines():
        if any(kw in line.lower() for kw in ["ethernet", "network", "wireless", "infiniband"]):
            m = re.search(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]", line)
            desc = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            devices.append({
                "description": desc,
                "pci_id": m.group(1) if m else "",
                "type": "network",
            })
    return devices


def detect_pci_storage_devices() -> list[dict]:
    lspci = _run(["lspci", "-nn"])
    devices: list[dict] = []
    for line in lspci.splitlines():
        if any(kw in line.lower() for kw in ["sata", "nvme", "ahci", "storage"]):
            m = re.search(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]", line)
            desc = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            devices.append({
                "description": desc,
                "pci_id": m.group(1) if m else "",
                "type": "storage",
            })
    return devices


def detect_all_pci_devices() -> list[dict]:
    return detect_pci_network_devices() + detect_pci_storage_devices()


# ── Memory Bandwidth Bench (sysbench) ──

def benchmark_memory_bandwidth(duration_sec: int = 5) -> dict[str, float]:
    result: dict[str, float] = {
        "read_ops_per_sec": 0.0,
        "write_ops_per_sec": 0.0,
        "total_ops_per_sec": 0.0,
        "read_mib_per_sec": 0.0,
        "write_mib_per_sec": 0.0,
    }
    out = _run(["sysbench", "memory", "--memory-oper=read", f"--time={duration_sec}", "run"], timeout=duration_sec + 10)
    if out:
        for line in out.splitlines():
            if "MiB/sec" in line:
                m = re.search(r"([\d.]+)\s*MiB/sec", line)
                if m:
                    result["read_mib_per_sec"] = float(m.group(1))
            if "Total operations:" in line:
                m = re.search(r"([\d.]+)", line)
                if m:
                    result["read_ops_per_sec"] = float(m.group(1)) / max(duration_sec, 1)
    out = _run(["sysbench", "memory", "--memory-oper=write", f"--time={duration_sec}", "run"], timeout=duration_sec + 10)
    if out:
        for line in out.splitlines():
            if "MiB/sec" in line:
                m = re.search(r"([\d.]+)\s*MiB/sec", line)
                if m:
                    result["write_mib_per_sec"] = float(m.group(1))
            if "Total operations:" in line:
                m = re.search(r"([\d.]+)", line)
                if m:
                    result["write_ops_per_sec"] = float(m.group(1)) / max(duration_sec, 1)
    result["total_ops_per_sec"] = round(result["read_ops_per_sec"] + result["write_ops_per_sec"], 1)
    result["read_mib_per_sec"] = round(result["read_mib_per_sec"], 1)
    result["write_mib_per_sec"] = round(result["write_mib_per_sec"], 1)
    return result


# ── Remote Node Detection ──

def detect_remote_hardware(host: str, key_file: str | None = None) -> dict[str, Any]:
    lscpu = _run_remote(host, ["lscpu"], key_file=key_file)
    dmi = _run_remote(host, ["dmidecode", "-t", "memory"], timeout=15, key_file=key_file)
    lspci = _run_remote(host, ["lspci", "-nn"], key_file=key_file)
    nvsmi = _run_remote(host, ["nvidia-smi", "--query-gpu=index,name,memory.total",
                                 "--format=csv,noheader,nounits"], timeout=15, key_file=key_file)
    return {
        "hostname": host,
        "lscpu_output": lscpu,
        "dmidecode_output": dmi,
        "lspci_output": lspci,
        "nvidia_smi_output": nvsmi,
    }


# ── Container Awareness ──

def is_running_in_container() -> bool:
    cgroup = "/proc/1/cgroup"
    if os.path.isfile(cgroup):
        try:
            with open(cgroup) as f:
                content = f.read()
            if "docker" in content or "kubepods" in content or "containerd" in content:
                return True
        except (OSError, FileNotFoundError):
            pass
    if os.path.isfile("/.dockerenv"):
        return True
    return False


def detect_container_limits() -> dict[str, Any]:
    limits: dict[str, Any] = {"in_container": is_running_in_container()}
    if not limits["in_container"]:
        return limits
    cpuset = "/sys/fs/cgroup/cpuset/cpuset.cpus"
    if os.path.isfile(cpuset):
        try:
            with open(cpuset) as f:
                limits["cpuset_cpus"] = f.read().strip()
        except OSError:
            pass
    cpu_max = "/sys/fs/cgroup/cpu.max"
    if os.path.isfile(cpu_max):
        try:
            with open(cpu_max) as f:
                limits["cpu_max"] = f.read().strip()
        except OSError:
            pass
    mem_max = "/sys/fs/cgroup/memory.max"
    if os.path.isfile(mem_max):
        try:
            with open(mem_max) as f:
                val = f.read().strip()
                if val and val != "max":
                    limits["memory_max_bytes"] = int(val)
                    limits["memory_max_gb"] = round(int(val) / (1024**3), 1)
        except (OSError, ValueError):
            pass
    return limits


# ── Unified Hardware Detection ──

def detect_local_hardware() -> dict[str, Any]:
    cpu = detect_local_cpu()
    memory = detect_local_memory()
    gpu_pci_ids = detect_local_gpu_pci_ids()
    gpus = detect_all_gpus()
    numa = detect_numa_topology()
    bios = detect_bios_info()
    motherboard = detect_motherboard_info()
    network = detect_pci_network_devices()
    storage = detect_pci_storage_devices()
    container = detect_container_limits()

    mem_bw: dict[str, float] = {}
    if _run(["which", "sysbench"]):
        mem_bw = benchmark_memory_bandwidth()

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "in_container": container.get("in_container", False),
        "container_limits": container,
        "cpu": cpu,
        "numa": numa,
        "memory": memory.to_dict(),
        "memory_bandwidth_bench": mem_bw,
        "gpus": gpus,
        "gpu_pci_ids": gpu_pci_ids,
        "bios": bios,
        "motherboard": motherboard,
        "network_devices": network,
        "storage_devices": storage,
    }
