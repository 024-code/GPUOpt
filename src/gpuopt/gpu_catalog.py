from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GpuVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"


class GpuSegment(Enum):
    CONSUMER = "consumer"
    WORKSTATION = "workstation"
    DATA_CENTER = "data_center"
    ENTRY = "entry"


@dataclass
class GpuCapabilities:
    ray_tracing: bool = False
    tensor_cores: bool = False
    dlss: str | None = None
    fsr: str | None = None
    xess: str | None = None
    av1_encode: bool = False
    av1_decode: bool = False
    pcie_gen: int = 4
    nvlink: bool = False
    ecc_memory: bool = False
    display_outputs: bool = True
    dlss_frame_gen: bool = False
    reflex: bool = False
    broadcast: bool = False
    dp21: bool = False
    dpas_int4: bool = False
    multi_frame_gen: bool = False

    def summary(self) -> list[str]:
        tags: list[str] = []
        if self.ray_tracing:
            tags.append("ray_tracing")
        if self.tensor_cores:
            tags.append("tensor_cores")
        if self.dlss:
            tags.append(f"dlss_{self.dlss}")
        if self.fsr:
            tags.append(f"fsr_{self.fsr}")
        if self.xess:
            tags.append(f"xess_{self.xess}")
        if self.av1_encode:
            tags.append("av1_encode")
        if self.av1_decode:
            tags.append("av1_decode")
        if self.nvlink:
            tags.append("nvlink")
        if self.ecc_memory:
            tags.append("ecc")
        if self.dp21:
            tags.append("dp21")
        if self.multi_frame_gen:
            tags.append("multi_frame_gen")
        return tags


@dataclass
class GpuCatalogEntry:
    vendor: GpuVendor
    segment: GpuSegment
    model_full: str
    model_short: str
    aliases: list[str] = field(default_factory=list)
    vram_gib: float = 0.0
    vram_type: str = ""
    memory_bus_bits: int = 0
    memory_bandwidth_gbps: float = 0.0
    tdp_watts: float = 0.0
    base_clock_mhz: float = 0.0
    boost_clock_mhz: float = 0.0
    tensor_tflops_fp16: float = 0.0
    tensor_tflops_fp8: float = 0.0
    shader_tflops_fp32: float = 0.0
    capabilities: GpuCapabilities = field(default_factory=GpuCapabilities)

    def matches(self, name: str) -> bool:
        n = name.lower().strip().replace("-", " ").replace("_", " ")
        if self.model_full.lower().replace("-", " ") == n:
            return True
        if self.model_short.lower().replace("-", " ") == n:
            return True
        for alias in self.aliases:
            if alias.lower().replace("-", " ") == n:
                return True
        for alias in self.aliases + [self.model_short]:
            if alias.lower().replace("-", " ") in n:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor.value,
            "segment": self.segment.value,
            "model_full": self.model_full,
            "model_short": self.model_short,
            "aliases": self.aliases,
            "vram_gib": self.vram_gib,
            "vram_type": self.vram_type,
            "memory_bus_bits": self.memory_bus_bits,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "tdp_watts": self.tdp_watts,
            "base_clock_mhz": self.base_clock_mhz,
            "boost_clock_mhz": self.boost_clock_mhz,
            "tensor_tflops_fp16": self.tensor_tflops_fp16,
            "tensor_tflops_fp8": self.tensor_tflops_fp8,
            "shader_tflops_fp32": self.shader_tflops_fp32,
            "capabilities": self.capabilities.summary(),
        }


def _nvidia(name: str, short: str, vram: float, vram_type: str, bus: int,
            tdp: float, base: float, boost: float, tf16: float, tf8: float,
            shader: float, pcie: int = 4, aliases: list[str] | None = None,
            rt: bool = True, tensor: bool = True, dlss: str = "3.5",
            av1: bool = True, ecc: bool = False, nvlink: bool = False,
            dp21: bool = False, mfg: bool = False,
            seg: GpuSegment = GpuSegment.CONSUMER, bw: float = 0.0,
            display: bool | None = None) -> GpuCatalogEntry:
    disp = display if display is not None else (seg != GpuSegment.DATA_CENTER)
    return GpuCatalogEntry(
        vendor=GpuVendor.NVIDIA, segment=seg,
        model_full=name, model_short=short, aliases=aliases or [],
        vram_gib=vram, vram_type=vram_type, memory_bus_bits=bus,
        memory_bandwidth_gbps=bw, tdp_watts=tdp,
        base_clock_mhz=base, boost_clock_mhz=boost,
        tensor_tflops_fp16=tf16, tensor_tflops_fp8=tf8,
        shader_tflops_fp32=shader,
        capabilities=GpuCapabilities(
            ray_tracing=rt, tensor_cores=tensor, dlss=dlss,
            av1_encode=av1, av1_decode=av1, pcie_gen=pcie,
            ecc_memory=ecc, nvlink=nvlink, display_outputs=disp,
            dlss_frame_gen=True, reflex=True, broadcast=True,
            dp21=dp21, multi_frame_gen=mfg,
        ),
    )


def _amd(name: str, short: str, vram: float, vram_type: str, bus: int,
         tdp: float, base: float, boost: float, shader: float, pcie: int = 4,
         aliases: list[str] | None = None, rt: bool = True, av1: bool = True,
         dp21: bool = False, ecc: bool = False,
         seg: GpuSegment = GpuSegment.CONSUMER, bw: float = 0.0,
         display: bool | None = None) -> GpuCatalogEntry:
    disp = display if display is not None else (seg != GpuSegment.DATA_CENTER)
    return GpuCatalogEntry(
        vendor=GpuVendor.AMD, segment=seg,
        model_full=name, model_short=short, aliases=aliases or [],
        vram_gib=vram, vram_type=vram_type, memory_bus_bits=bus,
        memory_bandwidth_gbps=bw, tdp_watts=tdp,
        base_clock_mhz=base, boost_clock_mhz=boost,
        shader_tflops_fp32=shader,
        capabilities=GpuCapabilities(
            ray_tracing=rt, fsr="3", av1_encode=av1, av1_decode=av1,
            pcie_gen=pcie, ecc_memory=ecc, dp21=dp21,
            display_outputs=disp,
        ),
    )


def _intel(name: str, short: str, vram: float, vram_type: str, bus: int,
           tdp: float, base: float, boost: float, shader: float, pcie: int = 4,
           aliases: list[str] | None = None, rt: bool = True,
           seg: GpuSegment = GpuSegment.CONSUMER, bw: float = 0.0) -> GpuCatalogEntry:
    return GpuCatalogEntry(
        vendor=GpuVendor.INTEL, segment=seg,
        model_full=name, model_short=short, aliases=aliases or [],
        vram_gib=vram, vram_type=vram_type, memory_bus_bits=bus,
        memory_bandwidth_gbps=bw, tdp_watts=tdp,
        base_clock_mhz=base, boost_clock_mhz=boost,
        shader_tflops_fp32=shader,
        capabilities=GpuCapabilities(
            ray_tracing=rt, xess="1.4", av1_encode=True, av1_decode=True,
            pcie_gen=pcie,
        ),
    )


GPU_CATALOG: list[GpuCatalogEntry] = [
    # ── NVIDIA GeForce RTX 5000 (Blackwell) ──
    _nvidia("NVIDIA GeForce RTX 5090", "RTX 5090", 32, "GDDR7", 512, 575, 2010, 2520, 1670, 3340, 104,
            pcie=5, dlss="4", mfg=True, aliases=["rtx5090", "5090"], bw=1792),
    _nvidia("NVIDIA GeForce RTX 5080", "RTX 5080", 16, "GDDR7", 256, 360, 2295, 2610, 900, 1800, 56,
            pcie=5, dlss="4", mfg=True, aliases=["rtx5080", "5080"], bw=960),
    _nvidia("NVIDIA GeForce RTX 5070 Ti", "RTX 5070 Ti", 16, "GDDR7", 256, 300, 2295, 2475, 730, 1460, 46,
            pcie=5, dlss="4", mfg=True, aliases=["rtx5070ti", "5070ti", "rtx 5070 ti"], bw=960),
    _nvidia("NVIDIA GeForce RTX 5070", "RTX 5070", 12, "GDDR7", 192, 250, 2160, 2520, 610, 1220, 38,
            pcie=5, dlss="4", mfg=True, aliases=["rtx5070", "5070"], bw=672),
    _nvidia("NVIDIA GeForce RTX 5060 Ti", "RTX 5060 Ti", 16, "GDDR7", 128, 180, 2250, 2500, 420, 840, 26,
            pcie=5, dlss="4", mfg=True, aliases=["rtx5060ti", "5060ti", "rtx 5060 ti"], bw=448),
    _nvidia("NVIDIA GeForce RTX 5060", "RTX 5060", 8, "GDDR7", 128, 150, 2200, 2450, 350, 700, 22,
            pcie=5, dlss="4", mfg=True, aliases=["rtx5060", "5060"], bw=448),

    # ── NVIDIA GeForce RTX 4000 (Ada Lovelace) ──
    _nvidia("NVIDIA GeForce RTX 4090", "RTX 4090", 24, "GDDR6X", 384, 450, 2235, 2520, 330, 660, 83,
            aliases=["rtx4090", "4090"], bw=1008),
    _nvidia("NVIDIA GeForce RTX 4080 SUPER", "RTX 4080 SUPER", 16, "GDDR6X", 256, 320, 2295, 2550, 220, 440, 52,
            aliases=["rtx4080super", "4080super", "rtx 4080 super"], bw=736),
    _nvidia("NVIDIA GeForce RTX 4070 Ti SUPER", "RTX 4070 Ti SUPER", 16, "GDDR6X", 256, 285, 2340, 2610, 200, 400, 47,
            aliases=["rtx4070tisuper", "4070tisuper", "rtx 4070 ti super", "rtx 4070 ti"], bw=672),
    _nvidia("NVIDIA GeForce RTX 4070 SUPER", "RTX 4070 SUPER", 12, "GDDR6X", 192, 220, 1980, 2475, 158, 316, 36,
            aliases=["rtx4070super", "4070super", "rtx 4070 super"], bw=504),
    _nvidia("NVIDIA GeForce RTX 4070", "RTX 4070", 12, "GDDR6X", 192, 200, 1920, 2475, 145, 290, 29,
            aliases=["rtx4070", "4070"], bw=504),
    _nvidia("NVIDIA GeForce RTX 4060 Ti", "RTX 4060 Ti", 16, "GDDR6", 128, 165, 2310, 2535, 110, 220, 22,
            aliases=["rtx4060ti", "4060ti", "rtx 4060 ti", "nvidia geforce rtx 4060 ti 16gb"], bw=288),
    _nvidia("NVIDIA GeForce RTX 4060", "RTX 4060", 8, "GDDR6", 128, 115, 1830, 2460, 75, 150, 15,
            aliases=["rtx4060", "4060"], bw=272),
    _nvidia("NVIDIA GeForce RTX 3090 Ti", "RTX 3090 Ti", 24, "GDDR6X", 384, 450, 1560, 1860, 160, 320, 40,
            dlss="2.0", aliases=["rtx3090ti", "3090ti", "rtx 3090 ti"], bw=1008),
    _nvidia("NVIDIA GeForce RTX 3090", "RTX 3090", 24, "GDDR6X", 384, 350, 1395, 1695, 142, 284, 36,
            dlss="2.0", av1=False, aliases=["rtx3090", "3090"], bw=936),
    _nvidia("NVIDIA GeForce RTX 3080", "RTX 3080", 10, "GDDR6X", 320, 320, 1440, 1710, 119, 238, 30,
            dlss="2.0", av1=False, aliases=["rtx3080", "3080", "nvidia geforce rtx 3080 10gb"], bw=760),
    _nvidia("NVIDIA GeForce RTX 3070", "RTX 3070", 8, "GDDR6", 256, 220, 1500, 1725, 81, 162, 20,
            dlss="2.0", av1=False, aliases=["rtx3070", "3070"], bw=448),
    _nvidia("NVIDIA GeForce GTX 1650", "GTX 1650", 4, "GDDR5", 128, 75, 1485, 1665, 0, 0, 3,
            rt=False, tensor=False, dlss=None, av1=False,
            aliases=["gtx1650", "1650"], seg=GpuSegment.ENTRY, bw=128),
    _nvidia("NVIDIA GeForce GTX 1630", "GTX 1630", 4, "GDDR6", 64, 75, 1740, 1815, 0, 0, 2,
            rt=False, tensor=False, dlss=None, av1=False,
            aliases=["gtx1630", "1630"], seg=GpuSegment.ENTRY, bw=96),

    # ── AMD Radeon RX 7000 ──
    _amd("AMD Radeon RX 7900 XTX", "RX 7900 XTX", 24, "GDDR6", 384, 355, 1855, 2498, 61,
         dp21=True, aliases=["7900xtx", "rx7900xtx"], bw=960),
    _amd("AMD Radeon RX 7900 XT", "RX 7900 XT", 20, "GDDR6", 320, 315, 1500, 2394, 52,
         dp21=True, aliases=["7900xt", "rx7900xt"], bw=800),
    _amd("AMD Radeon RX 7900 GRE", "RX 7900 GRE", 16, "GDDR6", 256, 260, 1270, 2245, 37,
         aliases=["7900gre", "rx7900gre"], bw=576),
    _amd("AMD Radeon RX 7800 XT", "RX 7800 XT", 16, "GDDR6", 256, 263, 1295, 2430, 37,
         aliases=["7800xt", "rx7800xt"], bw=624),
    _amd("AMD Radeon RX 7700 XT", "RX 7700 XT", 12, "GDDR6", 192, 245, 1435, 2544, 35,
         aliases=["7700xt", "rx7700xt"], bw=432),
    _amd("AMD Radeon RX 7600 XT", "RX 7600 XT", 16, "GDDR6", 128, 190, 1720, 2755, 22,
         aliases=["7600xt", "rx7600xt"], bw=288),
    _amd("AMD Radeon RX 7600", "RX 7600", 8, "GDDR6", 128, 165, 1720, 2655, 22,
         aliases=["7600", "rx7600"], bw=288),

    # ── AMD Radeon RX 6000 ──
    _amd("AMD Radeon RX 6950 XT", "RX 6950 XT", 16, "GDDR6", 256, 335, 1925, 2310, 24,
         av1=False, aliases=["6950xt", "rx6950xt"], bw=576),
    _amd("AMD Radeon RX 6800 XT", "RX 6800 XT", 16, "GDDR6", 256, 300, 1825, 2250, 21,
         av1=False, aliases=["6800xt", "rx6800xt"], bw=512),
    _amd("AMD Radeon RX 6750 XT", "RX 6750 XT", 12, "GDDR6", 192, 250, 2150, 2600, 14,
         av1=False, aliases=["6750xt", "rx6750xt"], bw=432),
    _amd("AMD Radeon RX 6600", "RX 6600", 8, "GDDR6", 128, 132, 1626, 2491, 9,
         av1=False, aliases=["6600", "rx6600"], bw=224),

    # ── Intel Arc Alchemist ──
    _intel("Intel Arc A770", "Arc A770", 16, "GDDR6", 256, 225, 2100, 2400, 20,
           aliases=["a770", "arc a770", "intel arc a770 16gb"], seg=GpuSegment.CONSUMER, bw=560),
    _intel("Intel Arc A750", "Arc A750", 8, "GDDR6", 256, 225, 2050, 2400, 17,
           aliases=["a750", "arc a750"], seg=GpuSegment.CONSUMER, bw=512),
    _intel("Intel Arc A580", "Arc A580", 8, "GDDR6", 256, 185, 1700, 2400, 12,
           aliases=["a580", "arc a580"], seg=GpuSegment.CONSUMER, bw=512),
    _intel("Intel Arc A380", "Arc A380", 6, "GDDR6", 96, 75, 2000, 2450, 5,
           aliases=["a380", "arc a380"], seg=GpuSegment.ENTRY, bw=186),
    _intel("Intel Arc A310", "Arc A310", 4, "GDDR6", 64, 75, 2000, 2400, 3.5,
           aliases=["a310", "arc a310"], seg=GpuSegment.ENTRY, bw=124),

    # ── NVIDIA RTX Workstation (Ada) ──
    _nvidia("NVIDIA RTX 6000 Ada", "RTX 6000 Ada", 48, "GDDR6 ECC", 384, 300, 915, 2535, 280, 560, 65,
            ecc=True, aliases=["rtx6000ada", "6000ada", "nvidia rtx 6000 ada generation"],
            seg=GpuSegment.WORKSTATION, bw=960),
    _nvidia("NVIDIA RTX 5000 Ada", "RTX 5000 Ada", 32, "GDDR6 ECC", 256, 250, 900, 2520, 190, 380, 45,
            ecc=True, aliases=["rtx5000ada", "5000ada"],
            seg=GpuSegment.WORKSTATION, bw=640),
    _nvidia("NVIDIA RTX 4500 Ada", "RTX 4500 Ada", 24, "GDDR6 ECC", 192, 200, 850, 2475, 150, 300, 35,
            ecc=True, aliases=["rtx4500ada", "4500ada"],
            seg=GpuSegment.WORKSTATION, bw=480),
    _nvidia("NVIDIA RTX 4000 Ada SFF", "RTX 4000 Ada SFF", 20, "GDDR6", 160, 130, 800, 2200, 110, 220, 26,
            aliases=["rtx4000ada", "4000ada"],
            seg=GpuSegment.WORKSTATION, bw=416),

    # ── NVIDIA Ampere Workstation ──
    _nvidia("NVIDIA RTX A6000", "RTX A6000", 48, "GDDR6 ECC", 384, 300, 840, 1800, 160, 320, 40,
            ecc=True, dlss="2.0", av1=False, aliases=["a6000", "rtx a6000"],
            seg=GpuSegment.WORKSTATION, bw=768),
    _nvidia("NVIDIA RTX A5000", "RTX A5000", 24, "GDDR6 ECC", 384, 230, 840, 1800, 90, 180, 23,
            ecc=True, dlss="2.0", av1=False, aliases=["a5000", "rtx a5000"],
            seg=GpuSegment.WORKSTATION, bw=768),
    _nvidia("NVIDIA RTX A4000", "RTX A4000", 16, "GDDR6 ECC", 256, 140, 735, 1560, 60, 120, 15,
            ecc=True, dlss="2.0", av1=False, aliases=["a4000", "rtx a4000"],
            seg=GpuSegment.WORKSTATION, bw=448),
    _nvidia("NVIDIA RTX A2000", "RTX A2000", 12, "GDDR6", 192, 70, 562, 1200, 30, 60, 8,
            dlss="2.0", av1=False, aliases=["a2000", "rtx a2000"],
            seg=GpuSegment.WORKSTATION, bw=288),

    # ── AMD Radeon Pro ──
    _amd("AMD Radeon Pro W7900", "Radeon Pro W7900", 48, "GDDR6 ECC", 384, 295, 1350, 2495, 45,
         dp21=True, ecc=True, aliases=["w7900", "radeon pro w7900"],
         seg=GpuSegment.WORKSTATION, bw=864),
    _amd("AMD Radeon Pro W7800", "Radeon Pro W7800", 32, "GDDR6 ECC", 256, 260, 1200, 2400, 32,
         dp21=True, ecc=True, aliases=["w7800", "radeon pro w7800"],
         seg=GpuSegment.WORKSTATION, bw=576),
    _amd("AMD Radeon Pro W7700", "Radeon Pro W7700", 16, "GDDR6", 256, 190, 1100, 2200, 22,
         dp21=True, aliases=["w7700", "radeon pro w7700"],
         seg=GpuSegment.WORKSTATION, bw=512),
    _amd("AMD Radeon Pro W6800", "Radeon Pro W6800", 32, "GDDR6 ECC", 256, 250, 1200, 2075, 18,
         av1=False, ecc=True, aliases=["w6800", "radeon pro w6800"],
         seg=GpuSegment.WORKSTATION, bw=512),

    # ── Intel Arc Pro ──
    _intel("Intel Arc Pro A60", "Arc Pro A60", 12, "GDDR6", 192, 130, 1800, 2100, 8,
           aliases=["pro a60", "arc pro a60"], seg=GpuSegment.WORKSTATION, bw=384),
    _intel("Intel Arc Pro A50", "Arc Pro A50", 6, "GDDR6", 96, 75, 1700, 2000, 4.5,
           aliases=["pro a50", "arc pro a50"], seg=GpuSegment.WORKSTATION, bw=192),
    _intel("Intel Arc Pro A40", "Arc Pro A40", 6, "GDDR6", 96, 50, 1600, 1900, 3.5,
           aliases=["pro a40", "arc pro a40"], seg=GpuSegment.WORKSTATION, bw=192),

    # ── Data Center / AI GPUs ──
    _nvidia("NVIDIA H100-SXM-80GB", "H100", 80, "HBM3", 1024, 700, 1095, 1830, 990, 1979, 50,
            ecc=True, nvlink=True, aliases=["h100", "nvidia h100", "h100 sxm"],
            seg=GpuSegment.DATA_CENTER, display=False, bw=3072, pcie=5),
    _nvidia("NVIDIA H200-SXM-141GB", "H200", 141, "HBM3e", 1224, 700, 1095, 1830, 990, 1979, 50,
            ecc=True, nvlink=True, aliases=["h200", "nvidia h200", "h200 sxm"],
            seg=GpuSegment.DATA_CENTER, display=False, bw=4800, pcie=5),
    _nvidia("NVIDIA B200", "B200", 192, "HBM3e", 1224, 700, 1000, 1800, 1600, 3200, 60,
            ecc=True, nvlink=True, dlss=None,
            aliases=["b200", "nvidia b200", "blackwell b200"],
            seg=GpuSegment.DATA_CENTER, display=False, bw=4800, pcie=5, tensor=False),
    _nvidia("NVIDIA A100-SXM-80GB", "A100", 80, "HBM2e", 512, 400, 1095, 1410, 312, 624, 20,
            ecc=True, nvlink=True, dlss=None, av1=False,
            aliases=["a100", "nvidia a100", "a100 sxm"],
            seg=GpuSegment.DATA_CENTER, display=False, bw=2039, pcie=4),
    _nvidia("NVIDIA A100-SXM-40GB", "A100 40GB", 40, "HBM2e", 512, 250, 1095, 1410, 312, 624, 20,
            ecc=True, nvlink=True, dlss=None, av1=False,
            aliases=["a100 40gb", "a100-40gb"],
            seg=GpuSegment.DATA_CENTER, display=False, bw=1555, pcie=4),
    _nvidia("NVIDIA L40S", "L40S", 48, "GDDR6 ECC", 384, 300, 915, 2520, 280, 560, 65,
            ecc=True, aliases=["l40s", "nvidia l40s"],
            seg=GpuSegment.DATA_CENTER, bw=960, pcie=4),
    _nvidia("NVIDIA V100-SXM-32GB", "V100", 32, "HBM2", 4096, 300, 1246, 1380, 125, 0, 7.8,
            dlss=None, av1=False, ecc=True, nvlink=True,
            aliases=["v100", "nvidia v100", "v100 sxm"],
            seg=GpuSegment.DATA_CENTER, display=False, bw=900, pcie=3),
    _nvidia("NVIDIA T4", "T4", 16, "GDDR6", 256, 70, 585, 1590, 32, 0, 8.1,
            dlss=None, av1=False, rt=False,
            aliases=["t4", "nvidia t4", "tesla t4"],
            seg=GpuSegment.DATA_CENTER, bw=320, pcie=3),
    _nvidia("NVIDIA L4", "L4", 24, "GDDR6 ECC", 192, 72, 735, 1560, 60, 120, 15,
            ecc=True, aliases=["l4", "nvidia l4"],
            seg=GpuSegment.DATA_CENTER, bw=300, pcie=4),
    _amd("AMD Instinct MI300X", "MI300X", 192, "HBM3", 8192, 750, 1400, 2100, 130,
         ecc=True, aliases=["mi300x", "amd mi300x", "instinct mi300x"],
         seg=GpuSegment.DATA_CENTER, bw=5376),
    _amd("AMD Instinct MI250X", "MI250X", 128, "HBM2e", 8192, 500, 1375, 1700, 48,
         ecc=True, av1=False, aliases=["mi250x", "amd mi250x", "instinct mi250x"],
         seg=GpuSegment.DATA_CENTER, bw=3277),
]


class GpuCatalog:
    def __init__(self) -> None:
        self._entries = GPU_CATALOG
        self._by_short: dict[str, GpuCatalogEntry] = {}
        self._by_vendor: dict[str, list[GpuCatalogEntry]] = {}
        for e in self._entries:
            self._by_short[e.model_short.lower()] = e
            v = e.vendor.value
            if v not in self._by_vendor:
                self._by_vendor[v] = []
            self._by_vendor[v].append(e)

    @property
    def entries(self) -> list[GpuCatalogEntry]:
        return self._entries

    def lookup(self, name: str) -> GpuCatalogEntry | None:
        n = name.strip()
        if n.lower() in self._by_short:
            return self._by_short[n.lower()]
        for entry in self._entries:
            if entry.matches(n):
                return entry
        n_clean = n.lower().replace("nvidia ", "").replace("amd ", "").replace("intel ", "")
        n_clean = n_clean.replace("-", " ").replace("_", " ")
        words = n_clean.split()
        best_score = 0.0
        best_entry: GpuCatalogEntry | None = None
        for entry in self._entries:
            e_clean = entry.model_full.lower().replace("-", " ").replace("_", " ")
            e_short = entry.model_short.lower().replace("-", " ").replace("_", " ")
            for alias in entry.aliases:
                a_clean = alias.lower().replace("-", " ").replace("_", " ")
                score = sum(1 for w in words if w in e_clean or w in e_short or w in a_clean)
                score /= max(len(words), 1)
                if score > best_score:
                    best_score = score
                    best_entry = entry
        if best_score >= 0.4:
            return best_entry
        return None

    def query(self, vendor: str | None = None, segment: str | None = None,
              min_vram: float | None = None, capabilities: list[str] | None = None) -> list[dict]:
        results = self._entries
        if vendor:
            results = [e for e in results if e.vendor.value == vendor.lower()]
        if segment:
            results = [e for e in results if e.segment.value == segment.lower()]
        if min_vram:
            results = [e for e in results if e.vram_gib >= min_vram]
        if capabilities:
            for cap in capabilities:
                results = [e for e in results if cap in e.capabilities.summary()]
        return [e.to_dict() for e in results]

    def filter_by_capability(self, capability: str) -> list[GpuCatalogEntry]:
        return [e for e in self._entries if capability in e.capabilities.summary()]

    def get_training_capable(self) -> list[GpuCatalogEntry]:
        return [e for e in self._entries if e.tensor_tflops_fp16 >= 50 and e.vram_gib >= 16]

    def get_inference_capable(self) -> list[GpuCatalogEntry]:
        return [e for e in self._entries if e.shader_tflops_fp32 >= 10 and e.vram_gib >= 8]

    def group_by_vendor(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for v, entries in self._by_vendor.items():
            result[v] = [e.to_dict() for e in entries]
        return result

    def group_by_segment(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for e in self._entries:
            s = e.segment.value
            if s not in result:
                result[s] = []
            result[s].append(e.to_dict())
        return result


_catalog_instance: GpuCatalog | None = None


def get_gpu_catalog() -> GpuCatalog:
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = GpuCatalog()
    return _catalog_instance


def lookup_gpu(name: str) -> GpuCatalogEntry | None:
    return get_gpu_catalog().lookup(name)
