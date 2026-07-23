from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CpuVendor(Enum):
    INTEL = "intel"
    AMD = "amd"
    ARM = "arm"


class CpuSocket(Enum):
    LGA1151 = "lga1151"
    LGA1200 = "lga1200"
    LGA1700 = "lga1700"
    LGA1851 = "lga1851"
    LGA2011_3 = "lga2011_3"
    LGA2066 = "lga2066"
    LGA4189 = "lga4189"
    LGA4677 = "lga4677"
    LGA7529 = "lga7529"
    SP5 = "sp5"
    SP6 = "sp6"
    TRX50 = "trx50"
    sWRX8 = "swrx8"
    AM4 = "am4"
    AM5 = "am5"
    BGA2049 = "bga2049"
    BGA = "bga"
    ARM = "arm"


class IgpuVendor(Enum):
    INTEL = "intel"
    AMD = "amd"


class MediaCapability(Enum):
    AV1_DECODE = "av1_decode"
    AV1_ENCODE = "av1_encode"
    HEVC_DECODE = "hevc_decode"
    HEVC_ENCODE = "hevc_encode"
    AVC_DECODE = "avc_decode"
    AVC_ENCODE = "avc_encode"
    VP9_DECODE = "vp9_decode"
    VP9_ENCODE = "vp9_encode"


@dataclass
class IgpuSpec:
    model: str
    vendor: IgpuVendor
    execution_units: int = 0
    compute_units: int = 0
    max_clock_mhz: float = 0.0
    pci_device_id: str = ""
    memory_type: str = ""
    supports_av1_decode: bool = False
    supports_av1_encode: bool = False
    supports_hevc_decode: bool = True
    supports_hevc_encode: bool = True
    supports_avc_decode: bool = True
    supports_avc_encode: bool = True
    supports_vp9_decode: bool = False
    supports_vp9_encode: bool = False
    shader_tflops_fp32: float = 0.0

    def media_capabilities(self) -> list[str]:
        caps: list[str] = []
        if self.supports_av1_decode:
            caps.append("av1_decode")
        if self.supports_av1_encode:
            caps.append("av1_encode")
        if self.supports_hevc_decode:
            caps.append("hevc_decode")
        if self.supports_hevc_encode:
            caps.append("hevc_encode")
        if self.supports_avc_decode:
            caps.append("avc_decode")
        if self.supports_avc_encode:
            caps.append("avc_encode")
        if self.supports_vp9_decode:
            caps.append("vp9_decode")
        if self.supports_vp9_encode:
            caps.append("vp9_encode")
        return caps


@dataclass
class CpuCatalogEntry:
    vendor: CpuVendor
    model_full: str
    model_short: str
    aliases: list[str] = field(default_factory=list)
    socket: CpuSocket | str = ""
    cores: int = 0
    threads: int = 0
    hybrid_cores_p: int = 0
    hybrid_cores_e: int = 0
    base_clock_ghz: float = 0.0
    boost_clock_ghz: float = 0.0
    tdp_watts: float = 0.0
    pl1_watts: float = 0.0
    pl2_watts: float = 0.0
    igpu: IgpuSpec | None = None
    memory_support: str = ""
    max_memory_gb: int = 0
    l3_cache_mb: float = 0.0
    pcie_version: int = 4
    pcie_lanes: int = 20
    architecture: str = ""
    lithography_nm: int = 0
    ecc_support: bool = False

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
        d: dict[str, Any] = {
            "vendor": self.vendor.value,
            "model_full": self.model_full,
            "model_short": self.model_short,
            "aliases": self.aliases,
            "socket": self.socket.value if isinstance(self.socket, CpuSocket) else self.socket,
            "cores": self.cores,
            "threads": self.threads,
            "hybrid_cores_p": self.hybrid_cores_p,
            "hybrid_cores_e": self.hybrid_cores_e,
            "base_clock_ghz": self.base_clock_ghz,
            "boost_clock_ghz": self.boost_clock_ghz,
            "tdp_watts": self.tdp_watts,
            "pl1_watts": self.pl1_watts,
            "pl2_watts": self.pl2_watts,
            "memory_support": self.memory_support,
            "max_memory_gb": self.max_memory_gb,
            "l3_cache_mb": self.l3_cache_mb,
            "pcie_version": self.pcie_version,
            "pcie_lanes": self.pcie_lanes,
            "architecture": self.architecture,
            "lithography_nm": self.lithography_nm,
            "ecc_support": self.ecc_support,
        }
        if self.igpu:
            d["igpu"] = {
                "model": self.igpu.model,
                "vendor": self.igpu.vendor.value,
                "execution_units": self.igpu.execution_units,
                "compute_units": self.igpu.compute_units,
                "max_clock_mhz": self.igpu.max_clock_mhz,
                "pci_device_id": self.igpu.pci_device_id,
                "memory_type": self.igpu.memory_type,
                "shader_tflops_fp32": self.igpu.shader_tflops_fp32,
                "media_capabilities": self.igpu.media_capabilities(),
            }
        return d


def _intel(name: str, short: str, socket: CpuSocket | str, cores: int,
           threads: int, base: float, boost: float, tdp: float,
           arch: str = "", l3: float = 0.0, pcie_ver: int = 4,
           pcie_lanes: int = 20, mem: str = "", max_mem: int = 128,
           litho: int = 0, ecc: bool = False,
           aliases: list[str] | None = None,
           p_cores: int = 0, e_cores: int = 0,
           pl1: float = 0.0, pl2: float = 0.0,
           igpu_model: str = "", igpu_eus: int = 0, igpu_clock: float = 0.0,
           igpu_pci: str = "", igpu_tflops: float = 0.0,
           igpu_av1_decode: bool = False, igpu_av1_encode: bool = False,
           igpu_vp9_decode: bool = False) -> CpuCatalogEntry:
    igpu: IgpuSpec | None = None
    if igpu_model:
        igpu = IgpuSpec(
            model=igpu_model, vendor=IgpuVendor.INTEL,
            execution_units=igpu_eus, max_clock_mhz=igpu_clock,
            pci_device_id=igpu_pci, memory_type=mem,
            shader_tflops_fp32=igpu_tflops,
            supports_av1_decode=igpu_av1_decode,
            supports_av1_encode=igpu_av1_encode,
            supports_vp9_decode=igpu_vp9_decode,
        )
    return CpuCatalogEntry(
        vendor=CpuVendor.INTEL, model_full=name, model_short=short,
        aliases=aliases or [], socket=socket, cores=cores,
        threads=threads, base_clock_ghz=base, boost_clock_ghz=boost,
        hybrid_cores_p=p_cores, hybrid_cores_e=e_cores,
        tdp_watts=tdp, pl1_watts=pl1 or tdp, pl2_watts=pl2 or boost * 10,
        architecture=arch, l3_cache_mb=l3,
        pcie_version=pcie_ver, pcie_lanes=pcie_lanes,
        memory_support=mem, max_memory_gb=max_mem,
        lithography_nm=litho, ecc_support=ecc, igpu=igpu,
    )


def _amd(name: str, short: str, socket: CpuSocket | str, cores: int,
         threads: int, base: float, boost: float, tdp: float,
         arch: str = "", l3: float = 0.0, pcie_ver: int = 4,
         pcie_lanes: int = 24, mem: str = "", max_mem: int = 128,
         litho: int = 0, ecc: bool = False,
         aliases: list[str] | None = None,
         p_cores: int = 0, e_cores: int = 0,
         pl1: float = 0.0, pl2: float = 0.0,
         igpu_model: str = "", igpu_cus: int = 0, igpu_clock: float = 0.0,
         igpu_pci: str = "", igpu_tflops: float = 0.0,
         igpu_av1_decode: bool = False, igpu_av1_encode: bool = False) -> CpuCatalogEntry:
    igpu: IgpuSpec | None = None
    if igpu_model:
        igpu = IgpuSpec(
            model=igpu_model, vendor=IgpuVendor.AMD,
            compute_units=igpu_cus, max_clock_mhz=igpu_clock,
            pci_device_id=igpu_pci, memory_type=mem,
            shader_tflops_fp32=igpu_tflops,
            supports_av1_decode=igpu_av1_decode,
            supports_av1_encode=igpu_av1_encode,
        )
    return CpuCatalogEntry(
        vendor=CpuVendor.AMD, model_full=name, model_short=short,
        aliases=aliases or [], socket=socket, cores=cores,
        threads=threads, base_clock_ghz=base, boost_clock_ghz=boost,
        hybrid_cores_p=p_cores, hybrid_cores_e=e_cores,
        tdp_watts=tdp, pl1_watts=pl1 or tdp, pl2_watts=pl2 or boost * 10,
        architecture=arch, l3_cache_mb=l3,
        pcie_version=pcie_ver, pcie_lanes=pcie_lanes,
        memory_support=mem, max_memory_gb=max_mem,
        lithography_nm=litho, ecc_support=ecc, igpu=igpu,
    )


def _arm(name: str, short: str, cores: int, threads: int,
          base: float, boost: float, tdp: float,
          arch: str = "", l3: float = 0.0, mem: str = "", max_mem: int = 1024,
          litho: int = 0, ecc: bool = True,
          aliases: list[str] | None = None,
          pcie_ver: int = 4, pcie_lanes: int = 64) -> CpuCatalogEntry:
    return CpuCatalogEntry(
        vendor=CpuVendor.ARM, model_full=name, model_short=short,
        aliases=aliases or [], socket=CpuSocket.ARM, cores=cores,
        threads=threads, base_clock_ghz=base, boost_clock_ghz=boost,
        tdp_watts=tdp, architecture=arch, l3_cache_mb=l3,
        pcie_version=pcie_ver, pcie_lanes=pcie_lanes,
        memory_support=mem, max_memory_gb=max_mem,
        lithography_nm=litho, ecc_support=ecc,
    )


CPU_CATALOG: list[CpuCatalogEntry] = [
    # ═══════════════════════════════════════════════════════════
    # Intel 10th Gen — Comet Lake-S (LGA1200, DDR4-2933)
    # iGPU: UHD 630 / UHD 610 (Gen 9.5)
    # PCI: 0x9BC8/0x9BC5 (UHD 630), 0x9BA8 (UHD 610)
    # Media: HEVC 10/12-bit encode/decode, AVC encode/decode, VP9 decode
    # No AV1 decode, no AV1 encode
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Core i9-10900K", "i9-10900K", CpuSocket.LGA1200,
           10, 20, 3.7, 5.3, 125, arch="Comet Lake", l3=20, mem="DDR4-2933", max_mem=128,
           litho=14, aliases=["10900k", "i9 10900k"],
           igpu_model="UHD 630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x9BC8", igpu_tflops=0.461),
    _intel("Intel Core i9-10900", "i9-10900", CpuSocket.LGA1200,
           10, 20, 2.8, 5.2, 65, arch="Comet Lake", l3=20, mem="DDR4-2933", max_mem=128,
           litho=14, aliases=["10900"],
           igpu_model="UHD 630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x9BC8", igpu_tflops=0.461),
    _intel("Intel Core i7-10700K", "i7-10700K", CpuSocket.LGA1200,
           8, 16, 3.8, 5.1, 125, arch="Comet Lake", l3=16, mem="DDR4-2933", max_mem=128,
           litho=14, aliases=["10700k", "i7 10700k"],
           igpu_model="UHD 630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x9BC8", igpu_tflops=0.461),
    _intel("Intel Core i7-10700", "i7-10700", CpuSocket.LGA1200,
           8, 16, 2.9, 4.8, 65, arch="Comet Lake", l3=16, mem="DDR4-2933", max_mem=128,
           litho=14, aliases=["10700"],
           igpu_model="UHD 630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x9BC8", igpu_tflops=0.461),
    _intel("Intel Core i5-10600K", "i5-10600K", CpuSocket.LGA1200,
           6, 12, 4.1, 4.8, 125, arch="Comet Lake", l3=12, mem="DDR4-2666", max_mem=128,
           litho=14, aliases=["10600k", "i5 10600k"],
           igpu_model="UHD 630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x9BC8", igpu_tflops=0.461),
    _intel("Intel Core i5-10500", "i5-10500", CpuSocket.LGA1200,
           6, 12, 3.1, 4.5, 65, arch="Comet Lake", l3=12, mem="DDR4-2666", max_mem=128,
           litho=14, aliases=["10500"],
           igpu_model="UHD 630", igpu_eus=24, igpu_clock=1150,
           igpu_pci="0x9BC8", igpu_tflops=0.442),
    _intel("Intel Core i3-10100", "i3-10100", CpuSocket.LGA1200,
           4, 8, 3.6, 4.3, 65, arch="Comet Lake", l3=6, mem="DDR4-2666", max_mem=128,
           litho=14, aliases=["10100"],
           igpu_model="UHD 630", igpu_eus=24, igpu_clock=1100,
           igpu_pci="0x9BC8", igpu_tflops=0.422),

    # ═══════════════════════════════════════════════════════════
    # Intel 11th Gen — Rocket Lake-S (LGA1200, DDR4-3200)
    # iGPU: UHD 750 (Xe 32EU) / UHD 730 (Xe 24EU)
    # PCI: 0x4C8A (UHD 750), 0x4C8B (UHD 730)
    # Media: HEVC 10/12-bit, AVC, VP9 decode; AV1 not in HW
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Core i9-11900K", "i9-11900K", CpuSocket.LGA1200,
           8, 16, 3.5, 5.3, 125, arch="Rocket Lake", l3=16, mem="DDR4-3200", max_mem=128,
           litho=14, aliases=["11900k", "i9 11900k"], pcie_ver=4,
           igpu_model="UHD 750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Core i9-11900", "i9-11900", CpuSocket.LGA1200,
           8, 16, 2.5, 5.2, 65, arch="Rocket Lake", l3=16, mem="DDR4-3200", max_mem=128,
           litho=14, aliases=["11900"], pcie_ver=4,
           igpu_model="UHD 750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Core i7-11700K", "i7-11700K", CpuSocket.LGA1200,
           8, 16, 3.6, 5.0, 125, arch="Rocket Lake", l3=16, mem="DDR4-3200", max_mem=128,
           litho=14, aliases=["11700k", "i7 11700k"], pcie_ver=4,
           igpu_model="UHD 750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Core i7-11700", "i7-11700", CpuSocket.LGA1200,
           8, 16, 2.5, 4.9, 65, arch="Rocket Lake", l3=16, mem="DDR4-3200", max_mem=128,
           litho=14, aliases=["11700"], pcie_ver=4,
           igpu_model="UHD 750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Core i5-11600K", "i5-11600K", CpuSocket.LGA1200,
           6, 12, 3.9, 4.9, 125, arch="Rocket Lake", l3=12, mem="DDR4-3200", max_mem=128,
           litho=14, aliases=["11600k", "i5 11600k"], pcie_ver=4,
           igpu_model="UHD 750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Core i5-11500", "i5-11500", CpuSocket.LGA1200,
           6, 12, 2.7, 4.6, 65, arch="Rocket Lake", l3=12, mem="DDR4-3200", max_mem=128,
           litho=14, aliases=["11500"], pcie_ver=4,
           igpu_model="UHD 750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Core i5-11400", "i5-11400", CpuSocket.LGA1200,
           6, 12, 2.6, 4.4, 65, arch="Rocket Lake", l3=12, mem="DDR4-3200", max_mem=128,
           litho=14, aliases=["11400"], pcie_ver=4,
           igpu_model="UHD 730", igpu_eus=24, igpu_clock=1300,
           igpu_pci="0x4C8B", igpu_tflops=0.998),

    # ═══════════════════════════════════════════════════════════
    # Intel 12th Gen — Alder Lake-S (LGA1700, DDR5-4800 / DDR4-3200)
    # iGPU: UHD 770 (Xe 32EU) / UHD 730 (Xe 24EU) / UHD 710 (Xe 16EU)
    # PCI: 0x4680 (UHD 770), 0x4692 (UHD 730), 0x4693 (UHD 710)
    # Media: AV1 decode only (no encode), HEVC/AVC/VP9 decode/encode
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Core i9-12900KS", "i9-12900KS", CpuSocket.LGA1700,
           16, 24, 3.4, 5.5, 150, arch="Alder Lake", l3=30, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12900ks", "i9 12900ks"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i9-12900K", "i9-12900K", CpuSocket.LGA1700,
           16, 24, 3.2, 5.2, 125, arch="Alder Lake", l3=30, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12900k", "i9 12900k"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i7-12700K", "i7-12700K", CpuSocket.LGA1700,
           12, 20, 3.6, 5.0, 125, arch="Alder Lake", l3=25, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12700k", "i7 12700k"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i7-12700", "i7-12700", CpuSocket.LGA1700,
           12, 20, 2.1, 4.9, 65, arch="Alder Lake", l3=25, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12700"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-12600K", "i5-12600K", CpuSocket.LGA1700,
           10, 16, 3.7, 4.9, 125, arch="Alder Lake", l3=20, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12600k", "i5 12600k"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-12500", "i5-12500", CpuSocket.LGA1700,
           6, 12, 3.0, 4.6, 65, arch="Alder Lake", l3=18, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12500"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1450,
           igpu_pci="0x4680", igpu_tflops=1.49, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-12400", "i5-12400", CpuSocket.LGA1700,
           6, 12, 2.5, 4.4, 65, arch="Alder Lake", l3=18, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12400"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 730", igpu_eus=24, igpu_clock=1450,
           igpu_pci="0x4692", igpu_tflops=1.11, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i3-12100", "i3-12100", CpuSocket.LGA1700,
           4, 8, 3.3, 4.3, 60, arch="Alder Lake", l3=12, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["12100"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 730", igpu_eus=24, igpu_clock=1400,
           igpu_pci="0x4692", igpu_tflops=1.08, igpu_av1_decode=True,
           igpu_vp9_decode=True),

    # ═══════════════════════════════════════════════════════════
    # Intel 13th Gen — Raptor Lake-S (LGA1700, DDR5-5600 / DDR4-3200)
    # iGPU: UHD 770 / UHD 730 / UHD 710
    # PCI: same as 12th gen
    # Media: AV1 decode only, HEVC/AVC/VP9
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Core i9-13900KS", "i9-13900KS", CpuSocket.LGA1700,
           24, 32, 3.2, 6.0, 150, arch="Raptor Lake", l3=36, mem="DDR5-5600/DDR4-3200",
           max_mem=128, litho=10, aliases=["13900ks", "i9 13900ks"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1650,
           igpu_pci="0x4680", igpu_tflops=1.69, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i9-13900K", "i9-13900K", CpuSocket.LGA1700,
           24, 32, 3.0, 5.8, 125, arch="Raptor Lake", l3=36, mem="DDR5-5600/DDR4-3200",
           max_mem=128, litho=10, aliases=["13900k", "i9 13900k"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1650,
           igpu_pci="0x4680", igpu_tflops=1.69, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i9-13900", "i9-13900", CpuSocket.LGA1700,
           24, 32, 2.0, 5.6, 65, arch="Raptor Lake", l3=36, mem="DDR5-5600/DDR4-3200",
           max_mem=128, litho=10, aliases=["13900"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1650,
           igpu_pci="0x4680", igpu_tflops=1.69, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i7-13700K", "i7-13700K", CpuSocket.LGA1700,
           16, 24, 3.4, 5.4, 125, arch="Raptor Lake", l3=30, mem="DDR5-5600/DDR4-3200",
           max_mem=128, litho=10, aliases=["13700k", "i7 13700k"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1600,
           igpu_pci="0x4680", igpu_tflops=1.64, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i7-13700", "i7-13700", CpuSocket.LGA1700,
           16, 24, 2.1, 5.2, 65, arch="Raptor Lake", l3=30, mem="DDR5-5600/DDR4-3200",
           max_mem=128, litho=10, aliases=["13700"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1600,
           igpu_pci="0x4680", igpu_tflops=1.64, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-13600K", "i5-13600K", CpuSocket.LGA1700,
           14, 20, 3.5, 5.1, 125, arch="Raptor Lake", l3=24, mem="DDR5-5600/DDR4-3200",
           max_mem=128, litho=10, aliases=["13600k", "i5 13600k"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-13500", "i5-13500", CpuSocket.LGA1700,
           14, 20, 2.5, 4.8, 65, arch="Raptor Lake", l3=24, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["13500"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-13400", "i5-13400", CpuSocket.LGA1700,
           10, 16, 2.5, 4.6, 65, arch="Raptor Lake", l3=20, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["13400"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 730", igpu_eus=24, igpu_clock=1550,
           igpu_pci="0x4692", igpu_tflops=1.19, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i3-13100", "i3-13100", CpuSocket.LGA1700,
           4, 8, 3.4, 4.5, 60, arch="Raptor Lake", l3=12, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, aliases=["13100"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 730", igpu_eus=24, igpu_clock=1500,
           igpu_pci="0x4692", igpu_tflops=1.15, igpu_av1_decode=True,
           igpu_vp9_decode=True),

    # ═══════════════════════════════════════════════════════════
    # Intel 14th Gen — Raptor Lake Refresh (LGA1700, DDR5-5600/DDR4-3200)
    # iGPU: UHD 770 / UHD 730
    # PCI: 0xA780 (UHD 770), 0xA782 (UHD 730)
    # Media: AV1 decode only, HEVC/AVC/VP9
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Core i9-14900KS", "i9-14900KS", CpuSocket.LGA1700,
           24, 32, 3.2, 6.2, 150, arch="Raptor Lake Refresh", l3=36,
           mem="DDR5-5600/DDR4-3200", max_mem=192, litho=10,
           aliases=["14900ks", "i9 14900ks"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1650,
           igpu_pci="0xA780", igpu_tflops=1.69, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i9-14900K", "i9-14900K", CpuSocket.LGA1700,
           24, 32, 3.2, 6.0, 125, arch="Raptor Lake Refresh", l3=36,
           mem="DDR5-5600/DDR4-3200", max_mem=192, litho=10,
           aliases=["14900k", "i9 14900k"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1650,
           igpu_pci="0xA780", igpu_tflops=1.69, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i9-14900", "i9-14900", CpuSocket.LGA1700,
           24, 32, 2.0, 5.8, 65, arch="Raptor Lake Refresh", l3=36,
           mem="DDR5-5600/DDR4-3200", max_mem=192, litho=10,
           aliases=["14900"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1650,
           igpu_pci="0xA780", igpu_tflops=1.69, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i7-14700K", "i7-14700K", CpuSocket.LGA1700,
           20, 28, 3.4, 5.6, 125, arch="Raptor Lake Refresh", l3=33,
           mem="DDR5-5600/DDR4-3200", max_mem=192, litho=10,
           aliases=["14700k", "i7 14700k"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1600,
           igpu_pci="0xA780", igpu_tflops=1.64, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i7-14700", "i7-14700", CpuSocket.LGA1700,
           20, 28, 2.1, 5.4, 65, arch="Raptor Lake Refresh", l3=33,
           mem="DDR5-5600/DDR4-3200", max_mem=192, litho=10,
           aliases=["14700"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1600,
           igpu_pci="0xA780", igpu_tflops=1.64, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-14600K", "i5-14600K", CpuSocket.LGA1700,
           14, 20, 3.5, 5.3, 125, arch="Raptor Lake Refresh", l3=24,
           mem="DDR5-5600/DDR4-3200", max_mem=192, litho=10,
           aliases=["14600k", "i5 14600k"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0xA780", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-14500", "i5-14500", CpuSocket.LGA1700,
           14, 20, 2.6, 5.0, 65, arch="Raptor Lake Refresh", l3=24,
           mem="DDR5-4800/DDR4-3200", max_mem=192, litho=10,
           aliases=["14500"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0xA780", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i5-14400", "i5-14400", CpuSocket.LGA1700,
           10, 16, 2.5, 4.7, 65, arch="Raptor Lake Refresh", l3=20,
           mem="DDR5-4800/DDR4-3200", max_mem=192, litho=10,
           aliases=["14400"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 730", igpu_eus=24, igpu_clock=1550,
           igpu_pci="0xA782", igpu_tflops=1.19, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core i3-14100", "i3-14100", CpuSocket.LGA1700,
           4, 8, 3.5, 4.7, 60, arch="Raptor Lake Refresh", l3=12,
           mem="DDR5-4800/DDR4-3200", max_mem=192, litho=10,
           aliases=["14100"], pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 730", igpu_eus=24, igpu_clock=1500,
           igpu_pci="0xA782", igpu_tflops=1.15, igpu_av1_decode=True,
           igpu_vp9_decode=True),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon E-2300 (Rocket Lake-E, LGA1200, DDR4-3200)
    # iGPU: UHD P750 (Xe 32EU)
    # PCI: 0x4C8A
    # Media: Same as Rocket Lake — no AV1 decode
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon E-2388G", "E-2388G", CpuSocket.LGA1200,
           8, 16, 3.2, 5.1, 95, arch="Rocket Lake-E", l3=16, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon e-2388g", "e2388g"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Xeon E-2378G", "E-2378G", CpuSocket.LGA1200,
           8, 16, 2.8, 5.1, 95, arch="Rocket Lake-E", l3=16, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon e-2378g", "e2378g"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Xeon E-2386G", "E-2386G", CpuSocket.LGA1200,
           6, 12, 3.5, 5.1, 95, arch="Rocket Lake-E", l3=12, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon e-2386g", "e2386g"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Xeon E-2378", "E-2378", CpuSocket.LGA1200,
           8, 16, 2.6, 4.8, 65, arch="Rocket Lake-E", l3=16, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon e-2378", "e2378"],
           pcie_ver=4, pcie_lanes=20),
    _intel("Intel Xeon E-2314", "E-2314", CpuSocket.LGA1200,
           4, 4, 2.8, 4.5, 65, arch="Rocket Lake-E", l3=8, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon e-2314", "e2314"],
           pcie_ver=4, pcie_lanes=20),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon W-1300 (Rocket Lake-W, LGA1200, DDR4-3200)
    # iGPU: UHD P750 (32EU)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon W-1390P", "W-1390P", CpuSocket.LGA1200,
           8, 16, 3.5, 5.3, 125, arch="Rocket Lake-W", l3=16, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon w-1390p", "w1390p"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Xeon W-1390", "W-1390", CpuSocket.LGA1200,
           8, 16, 2.8, 5.2, 80, arch="Rocket Lake-W", l3=16, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon w-1390", "w1390"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Xeon W-1370P", "W-1370P", CpuSocket.LGA1200,
           8, 16, 3.6, 5.2, 125, arch="Rocket Lake-W", l3=16, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon w-1370p", "w1370p"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Xeon W-1350P", "W-1350P", CpuSocket.LGA1200,
           6, 12, 4.0, 5.1, 125, arch="Rocket Lake-W", l3=12, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon w-1350p", "w1350p"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),
    _intel("Intel Xeon W-1350", "W-1350", CpuSocket.LGA1200,
           6, 12, 3.3, 5.0, 80, arch="Rocket Lake-W", l3=12, mem="DDR4-3200",
           max_mem=128, litho=14, ecc=True, aliases=["xeon w-1350", "w1350"],
           pcie_ver=4, pcie_lanes=20,
           igpu_model="UHD P750", igpu_eus=32, igpu_clock=1300,
           igpu_pci="0x4C8A", igpu_tflops=1.33),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon W-1500 (Alder Lake-W, LGA1700, DDR5-4800/DDR4-3200)
    # iGPU: UHD 770 (32EU)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon W-1550P", "W-1550P", CpuSocket.LGA1700,
           16, 24, 3.5, 5.3, 150, arch="Alder Lake-W", l3=30, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, ecc=True, aliases=["xeon w-1550p", "w1550p"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),
    _intel("Intel Xeon W-1550", "W-1550", CpuSocket.LGA1700,
           16, 24, 2.1, 5.0, 65, arch="Alder Lake-W", l3=30, mem="DDR5-4800/DDR4-3200",
           max_mem=128, litho=10, ecc=True, aliases=["xeon w-1550", "w1550"],
           pcie_ver=5, pcie_lanes=20,
           igpu_model="UHD 770", igpu_eus=32, igpu_clock=1550,
           igpu_pci="0x4680", igpu_tflops=1.59, igpu_av1_decode=True,
           igpu_vp9_decode=True),

    # ═══════════════════════════════════════════════════════════
    # AMD Ryzen 4000G Series — Renoir (AM4, DDR4-3200)
    # iGPU: Radeon Graphics (Vega)
    # PCI: 0x1636 (Vega 8/Cezanne), 0x1638 (Renoir)
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen 7 4700G", "Ryzen 7 4700G", CpuSocket.AM4,
         8, 16, 3.6, 4.4, 65, arch="Zen 2 Renoir", l3=8, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["4700g", "ryzen 7 4700g"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 8)", igpu_cus=8, igpu_clock=2100,
         igpu_pci="0x1636", igpu_tflops=2.15),
    _amd("AMD Ryzen 5 4600G", "Ryzen 5 4600G", CpuSocket.AM4,
         6, 12, 3.7, 4.2, 65, arch="Zen 2 Renoir", l3=8, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["4600g", "ryzen 5 4600g"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 7)", igpu_cus=7, igpu_clock=1900,
         igpu_pci="0x1636", igpu_tflops=1.70),
    _amd("AMD Ryzen 3 4300G", "Ryzen 3 4300G", CpuSocket.AM4,
         4, 8, 3.8, 4.0, 65, arch="Zen 2 Renoir", l3=4, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["4300g", "ryzen 3 4300g"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 5)", igpu_cus=5, igpu_clock=1700,
         igpu_pci="0x1636", igpu_tflops=1.09),

    # ═══════════════════════════════════════════════════════════
    # AMD Ryzen 5000G Series — Cezanne (AM4, DDR4-3200)
    # iGPU: Radeon Graphics (Vega 8 / Vega 7 / Vega 6)
    # PCI: 0x1638 (Cezanne)
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen 7 5700G", "Ryzen 7 5700G", CpuSocket.AM4,
         8, 16, 3.8, 4.6, 65, arch="Zen 3 Cezanne", l3=16, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["5700g", "ryzen 7 5700g"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 8)", igpu_cus=8, igpu_clock=2000,
         igpu_pci="0x1638", igpu_tflops=2.05),
    _amd("AMD Ryzen 5 5600GT", "Ryzen 5 5600GT", CpuSocket.AM4,
         6, 12, 3.6, 4.6, 65, arch="Zen 3 Cezanne", l3=16, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["5600gt", "ryzen 5 5600gt"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 7)", igpu_cus=7, igpu_clock=1900,
         igpu_pci="0x1638", igpu_tflops=1.70),
    _amd("AMD Ryzen 5 5600G", "Ryzen 5 5600G", CpuSocket.AM4,
         6, 12, 3.9, 4.4, 65, arch="Zen 3 Cezanne", l3=16, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["5600g", "ryzen 5 5600g"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 7)", igpu_cus=7, igpu_clock=1900,
         igpu_pci="0x1638", igpu_tflops=1.70),
    _amd("AMD Ryzen 5 5500GT", "Ryzen 5 5500GT", CpuSocket.AM4,
         6, 12, 3.6, 4.4, 65, arch="Zen 3 Cezanne", l3=16, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["5500gt", "ryzen 5 5500gt"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 7)", igpu_cus=7, igpu_clock=1900,
         igpu_pci="0x1638", igpu_tflops=1.70),
    _amd("AMD Ryzen 3 5300G", "Ryzen 3 5300G", CpuSocket.AM4,
         4, 8, 4.0, 4.2, 65, arch="Zen 3 Cezanne", l3=8, mem="DDR4-3200", max_mem=64,
         litho=7, aliases=["5300g", "ryzen 3 5300g"], pcie_ver=3, pcie_lanes=24,
         igpu_model="Radeon Graphics (Vega 6)", igpu_cus=6, igpu_clock=1700,
         igpu_pci="0x1638", igpu_tflops=1.31),

    # ═══════════════════════════════════════════════════════════
    # AMD Ryzen 8000G Series — Phoenix (AM5, DDR5-5200)
    # iGPU: Radeon 780M (RDNA 3, 12CU) / 760M (8CU) / 740M (4CU)
    # PCI: 0x15BF (780M), 0x15BB (760M), 0x15BD (740M)
    # Media: AV1 encode/decode, HEVC encode/decode, AVC, VP9 decode
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen 7 8700G", "Ryzen 7 8700G", CpuSocket.AM5,
         8, 16, 4.2, 5.1, 65, arch="Zen 4 Phoenix", l3=16, mem="DDR5-5200", max_mem=256,
         litho=4, aliases=["8700g", "ryzen 7 8700g"], pcie_ver=4, pcie_lanes=24,
         igpu_model="Radeon 780M", igpu_cus=12, igpu_clock=2900,
         igpu_pci="0x15BF", igpu_tflops=8.9, igpu_av1_decode=True, igpu_av1_encode=True),
    _amd("AMD Ryzen 5 8600G", "Ryzen 5 8600G", CpuSocket.AM5,
         6, 12, 4.3, 5.0, 65, arch="Zen 4 Phoenix", l3=16, mem="DDR5-5200", max_mem=256,
         litho=4, aliases=["8600g", "ryzen 5 8600g"], pcie_ver=4, pcie_lanes=24,
         igpu_model="Radeon 760M", igpu_cus=8, igpu_clock=2800,
         igpu_pci="0x15BB", igpu_tflops=5.7, igpu_av1_decode=True, igpu_av1_encode=True),
    _amd("AMD Ryzen 5 8500G", "Ryzen 5 8500G", CpuSocket.AM5,
         6, 12, 3.5, 5.0, 65, arch="Zen 4 Phoenix 2", l3=16, mem="DDR5-5200", max_mem=256,
         litho=4, aliases=["8500g", "ryzen 5 8500g"], pcie_ver=4, pcie_lanes=24,
         igpu_model="Radeon 740M", igpu_cus=4, igpu_clock=2800,
         igpu_pci="0x15BD", igpu_tflops=2.9, igpu_av1_decode=True, igpu_av1_encode=True),
    _amd("AMD Ryzen 3 8300G", "Ryzen 3 8300G", CpuSocket.AM5,
         4, 8, 3.4, 4.9, 65, arch="Zen 4 Phoenix 2", l3=8, mem="DDR5-5200", max_mem=256,
         litho=4, aliases=["8300g", "ryzen 3 8300g"], pcie_ver=4, pcie_lanes=24,
         igpu_model="Radeon 740M", igpu_cus=4, igpu_clock=2600,
         igpu_pci="0x15BD", igpu_tflops=2.7, igpu_av1_decode=True, igpu_av1_encode=True),

    # ═══════════════════════════════════════════════════════════
    # AMD Ryzen 7000G Series — Phoenix (AM5, DDR5-5200)
    # Same iGPU as 8000G (Radeon 780M / 760M / 740M)
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen 7 7700G", "Ryzen 7 7700G", CpuSocket.AM5,
         8, 16, 3.8, 5.0, 65, arch="Zen 4 Phoenix", l3=16, mem="DDR5-5200", max_mem=256,
         litho=4, aliases=["7700g", "ryzen 7 7700g"], pcie_ver=4, pcie_lanes=24,
         igpu_model="Radeon 780M", igpu_cus=12, igpu_clock=2900,
         igpu_pci="0x15BF", igpu_tflops=8.9, igpu_av1_decode=True, igpu_av1_encode=True),
    _amd("AMD Ryzen 5 7600G", "Ryzen 5 7600G", CpuSocket.AM5,
         6, 12, 3.9, 5.0, 65, arch="Zen 4 Phoenix", l3=16, mem="DDR5-5200", max_mem=256,
         litho=4, aliases=["7600g", "ryzen 5 7600g"], pcie_ver=4, pcie_lanes=24,
         igpu_model="Radeon 760M", igpu_cus=8, igpu_clock=2800,
         igpu_pci="0x15BB", igpu_tflops=5.7, igpu_av1_decode=True, igpu_av1_encode=True),

    # ═══════════════════════════════════════════════════════════
    # AMD Ryzen 7000 Desktop — Raphael (AM5, DDR5-5200)
    # iGPU: Radeon Graphics (RDNA 2, 2CU) — basic display only
    # PCI: 0x1640
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen 9 7950X", "Ryzen 9 7950X", CpuSocket.AM5,
         16, 32, 4.5, 5.7, 170, arch="Zen 4 Raphael", l3=64, mem="DDR5-5200", max_mem=128,
         litho=5, aliases=["7950x", "ryzen 9 7950x"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD Ryzen 9 7900X", "Ryzen 9 7900X", CpuSocket.AM5,
         12, 24, 4.7, 5.6, 170, arch="Zen 4 Raphael", l3=64, mem="DDR5-5200", max_mem=128,
         litho=5, aliases=["7900x", "ryzen 9 7900x"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD Ryzen 7 7700X", "Ryzen 7 7700X", CpuSocket.AM5,
         8, 16, 4.5, 5.4, 105, arch="Zen 4 Raphael", l3=32, mem="DDR5-5200", max_mem=128,
         litho=5, aliases=["7700x", "ryzen 7 7700x"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD Ryzen 5 7600X", "Ryzen 5 7600X", CpuSocket.AM5,
         6, 12, 4.7, 5.3, 105, arch="Zen 4 Raphael", l3=32, mem="DDR5-5200", max_mem=128,
         litho=5, aliases=["7600x", "ryzen 5 7600x"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),

    # ═══════════════════════════════════════════════════════════
    # AMD EPYC 4004 Series (AM5, DDR5-5200)
    # iGPU: Radeon Graphics (RDNA 2, 2CU) — basic BMC/display
    # ═══════════════════════════════════════════════════════════
    _amd("AMD EPYC 4564P", "EPYC 4564P", CpuSocket.AM5,
         16, 32, 3.7, 5.1, 170, arch="Zen 4", l3=64, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4564p", "4564p"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD EPYC 4484PX", "EPYC 4484PX", CpuSocket.AM5,
         16, 32, 4.0, 5.1, 170, arch="Zen 4", l3=128, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4484px", "4484px"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD EPYC 4464P", "EPYC 4464P", CpuSocket.AM5,
         12, 24, 3.7, 5.1, 140, arch="Zen 4", l3=32, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4464p", "4464p"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD EPYC 4364P", "EPYC 4364P", CpuSocket.AM5,
         8, 16, 4.1, 5.1, 105, arch="Zen 4", l3=32, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4364p", "4364p"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD EPYC 4244P", "EPYC 4244P", CpuSocket.AM5,
         6, 12, 4.1, 5.1, 105, arch="Zen 4", l3=32, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4244p", "4244p"], pcie_ver=5, pcie_lanes=28),
    _amd("AMD EPYC 4124P", "EPYC 4124P", CpuSocket.AM5,
         4, 8, 4.1, 5.1, 105, arch="Zen 4", l3=32, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4124p", "4124p"], pcie_ver=5, pcie_lanes=28),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon Scalable 4th Gen — Sapphire Rapids (LGA4677, DDR5-4800)
    # No integrated GPU — server only
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon Platinum 8490H", "Platinum 8490H", CpuSocket.LGA4677,
           60, 120, 1.9, 3.5, 350, arch="Sapphire Rapids", l3=112.5,
           mem="DDR5-4800", max_mem=4096, litho=10, ecc=True, pcie_ver=5, pcie_lanes=80,
           aliases=["8490h", "xeon platinum 8490h"]),
    _intel("Intel Xeon Platinum 8480+", "Platinum 8480+", CpuSocket.LGA4677,
           56, 112, 2.0, 3.8, 350, arch="Sapphire Rapids", l3=105,
           mem="DDR5-4800", max_mem=4096, litho=10, ecc=True, pcie_ver=5, pcie_lanes=80,
           aliases=["8480+", "8480plus", "xeon platinum 8480+"]),
    _intel("Intel Xeon Gold 6438M", "Gold 6438M", CpuSocket.LGA4677,
           32, 64, 2.2, 3.9, 205, arch="Sapphire Rapids", l3=60,
           mem="DDR5-4800", max_mem=4096, litho=10, ecc=True, pcie_ver=5, pcie_lanes=80,
           aliases=["6438m", "xeon gold 6438m"]),
    _intel("Intel Xeon Silver 4410Y", "Silver 4410Y", CpuSocket.LGA4677,
           12, 24, 2.0, 3.9, 150, arch="Sapphire Rapids", l3=30,
           mem="DDR5-4400", max_mem=4096, litho=10, ecc=True, pcie_ver=5, pcie_lanes=80,
           aliases=["4410y", "xeon silver 4410y"]),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon Scalable 5th Gen — Emerald Rapids (LGA4677, DDR5-5600)
    # No integrated GPU
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon Platinum 8592+", "Platinum 8592+", CpuSocket.LGA4677,
           64, 128, 2.0, 3.9, 350, arch="Emerald Rapids", l3=320,
           mem="DDR5-5600", max_mem=4096, litho=7, ecc=True, pcie_ver=5, pcie_lanes=80,
           aliases=["8592+", "8592plus", "xeon platinum 8592+"]),
    _intel("Intel Xeon Platinum 8580", "Platinum 8580", CpuSocket.LGA4677,
           60, 120, 2.0, 4.0, 350, arch="Emerald Rapids", l3=300,
           mem="DDR5-5600", max_mem=4096, litho=7, ecc=True, pcie_ver=5, pcie_lanes=80,
           aliases=["8580", "xeon platinum 8580"]),
    _intel("Intel Xeon Gold 6548N", "Gold 6548N", CpuSocket.LGA4677,
           32, 64, 2.8, 4.1, 250, arch="Emerald Rapids", l3=60,
           mem="DDR5-5200", max_mem=4096, litho=7, ecc=True, pcie_ver=5, pcie_lanes=80,
           aliases=["6548n", "xeon gold 6548n"]),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon 6 — Granite Rapids (LGA7529, DDR5-6400)
    # No integrated GPU
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon 6980P", "Xeon 6980P", CpuSocket.LGA7529,
           128, 256, 2.0, 3.9, 500, arch="Granite Rapids", l3=504,
           mem="DDR5-6400", max_mem=6144, litho=5, ecc=True, pcie_ver=5, pcie_lanes=136,
           aliases=["6980p", "xeon 6980p"]),
    _intel("Intel Xeon 6979B", "Xeon 6979B", CpuSocket.LGA7529,
           96, 192, 2.1, 3.9, 400, arch="Granite Rapids", l3=384,
           mem="DDR5-6400", max_mem=6144, litho=5, ecc=True, pcie_ver=5, pcie_lanes=136,
           aliases=["6979b", "xeon 6979b"]),

    # ═══════════════════════════════════════════════════════════
    # Intel Core Ultra 200S — Arrow Lake (LGA1851, DDR5-6400)
    # iGPU: Intel Xe-LPG (Alchemist+)
    # Media: AV1 decode/encode, HEVC, AVC, VP9
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Core Ultra 9 285K", "Ultra 9 285K", CpuSocket.LGA1851,
           24, 24, 3.7, 5.7, 125, arch="Arrow Lake", l3=36,
           mem="DDR5-6400", max_mem=192, litho=5, pcie_ver=5, pcie_lanes=24,
           aliases=["ultra 9 285k", "285k"], p_cores=8, e_cores=16, pl1=125, pl2=250,
           igpu_model="Intel Xe-LPG", igpu_eus=32, igpu_clock=2000,
           igpu_pci="0x7D45", igpu_tflops=2.05, igpu_av1_decode=True, igpu_av1_encode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core Ultra 7 265K", "Ultra 7 265K", CpuSocket.LGA1851,
           20, 20, 3.9, 5.5, 125, arch="Arrow Lake", l3=30,
           mem="DDR5-6400", max_mem=192, litho=5, pcie_ver=5, pcie_lanes=24,
           aliases=["ultra 7 265k", "265k"], p_cores=8, e_cores=12, pl1=125, pl2=250,
           igpu_model="Intel Xe-LPG", igpu_eus=32, igpu_clock=2000,
           igpu_pci="0x7D45", igpu_tflops=2.05, igpu_av1_decode=True, igpu_av1_encode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core Ultra 5 245K", "Ultra 5 245K", CpuSocket.LGA1851,
           14, 14, 4.2, 5.2, 125, arch="Arrow Lake", l3=24,
           mem="DDR5-6400", max_mem=192, litho=5, pcie_ver=5, pcie_lanes=24,
           aliases=["ultra 5 245k", "245k"], p_cores=6, e_cores=8, pl1=125, pl2=250,
           igpu_model="Intel Xe-LPG", igpu_eus=32, igpu_clock=1900,
           igpu_pci="0x7D45", igpu_tflops=1.95, igpu_av1_decode=True, igpu_av1_encode=True,
           igpu_vp9_decode=True),

    # ═══════════════════════════════════════════════════════════
    # Intel Core Ultra 100 — Meteor Lake (BGA2049, DDR5-5600/LPDDR5)
    # iGPU: Intel Arc (Xe-LPG)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Core Ultra 9 185H", "Ultra 9 185H", CpuSocket.BGA2049,
           16, 22, 2.3, 5.1, 45, arch="Meteor Lake", l3=24,
           mem="DDR5-5600/LPDDR5-7467", max_mem=96, litho=5, pcie_ver=5, pcie_lanes=28,
           aliases=["ultra 9 185h", "185h"], p_cores=6, e_cores=8,
           pl1=45, pl2=115,
           igpu_model="Intel Arc (8 Xe-Cores)", igpu_eus=128, igpu_clock=2350,
           igpu_pci="0x7D55", igpu_tflops=4.81, igpu_av1_decode=True, igpu_av1_encode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core Ultra 7 155H", "Ultra 7 155H", CpuSocket.BGA2049,
           16, 22, 1.4, 4.8, 28, arch="Meteor Lake", l3=24,
           mem="DDR5-5600/LPDDR5-7467", max_mem=96, litho=5, pcie_ver=5, pcie_lanes=28,
           aliases=["ultra 7 155h", "155h"], p_cores=6, e_cores=8,
           pl1=28, pl2=115,
           igpu_model="Intel Arc (8 Xe-Cores)", igpu_eus=128, igpu_clock=2250,
           igpu_pci="0x7D55", igpu_tflops=4.61, igpu_av1_decode=True, igpu_av1_encode=True,
           igpu_vp9_decode=True),
    _intel("Intel Core Ultra 5 125H", "Ultra 5 125H", CpuSocket.BGA2049,
           14, 18, 1.2, 4.5, 28, arch="Meteor Lake", l3=18,
           mem="DDR5-5600/LPDDR5-7467", max_mem=96, litho=5, pcie_ver=5, pcie_lanes=28,
           aliases=["ultra 5 125h", "125h"], p_cores=4, e_cores=8,
           pl1=28, pl2=115,
           igpu_model="Intel Arc (7 Xe-Cores)", igpu_eus=112, igpu_clock=2200,
           igpu_pci="0x7D55", igpu_tflops=3.94, igpu_av1_decode=True, igpu_av1_encode=True,
           igpu_vp9_decode=True),

    # ═══════════════════════════════════════════════════════════
    # AMD EPYC 9004 Series — Genoa (SP5, DDR5-4800)
    # No integrated GPU
    # ═══════════════════════════════════════════════════════════
    _amd("AMD EPYC 9654", "EPYC 9654", CpuSocket.SP5,
         96, 192, 2.4, 3.7, 360, arch="Zen 4 Genoa", l3=384,
         mem="DDR5-4800", max_mem=6144, litho=5, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["9654", "epyc 9654"]),
    _amd("AMD EPYC 9554", "EPYC 9554", CpuSocket.SP5,
         64, 128, 3.1, 3.75, 360, arch="Zen 4 Genoa", l3=256,
         mem="DDR5-4800", max_mem=6144, litho=5, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["9554", "epyc 9554"]),
    _amd("AMD EPYC 9374F", "EPYC 9374F", CpuSocket.SP5,
         32, 64, 3.85, 4.3, 320, arch="Zen 4 Genoa", l3=256,
         mem="DDR5-4800", max_mem=6144, litho=5, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["9374f", "epyc 9374f"]),
    _amd("AMD EPYC 9174F", "EPYC 9174F", CpuSocket.SP5,
         16, 32, 4.1, 4.4, 320, arch="Zen 4 Genoa", l3=64,
         mem="DDR5-4800", max_mem=6144, litho=5, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["9174f", "epyc 9174f"]),

    # ═══════════════════════════════════════════════════════════
    # AMD EPYC 9005 Series — Turin (SP5, DDR5-6000)
    # No integrated GPU
    # ═══════════════════════════════════════════════════════════
    _amd("AMD EPYC 9965", "EPYC 9965", CpuSocket.SP5,
         192, 384, 2.25, 3.7, 500, arch="Zen 5 Turin", l3=576,
         mem="DDR5-6000", max_mem=6144, litho=3, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["9965", "epyc 9965"]),
    _amd("AMD EPYC 9845", "EPYC 9845", CpuSocket.SP5,
         160, 320, 2.1, 3.7, 480, arch="Zen 5 Turin", l3=512,
         mem="DDR5-6000", max_mem=6144, litho=3, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["9845", "epyc 9845"]),

    # ═══════════════════════════════════════════════════════════
    # AMD Threadripper 7000 — Storm Peak (TRX50, DDR5-5200)
    # No integrated GPU
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen Threadripper 7980X", "TR 7980X", CpuSocket.TRX50,
         64, 128, 3.2, 5.1, 350, arch="Zen 4 Storm Peak", l3=256,
         mem="DDR5-5200", max_mem=512, litho=5, ecc=False, pcie_ver=5, pcie_lanes=48,
         aliases=["7980x", "tr 7980x", "threadripper 7980x"]),
    _amd("AMD Ryzen Threadripper 7970X", "TR 7970X", CpuSocket.TRX50,
         32, 64, 4.0, 5.3, 350, arch="Zen 4 Storm Peak", l3=128,
         mem="DDR5-5200", max_mem=512, litho=5, ecc=False, pcie_ver=5, pcie_lanes=48,
         aliases=["7970x", "tr 7970x", "threadripper 7970x"]),
    _amd("AMD Ryzen Threadripper 7960X", "TR 7960X", CpuSocket.TRX50,
         24, 48, 4.2, 5.3, 350, arch="Zen 4 Storm Peak", l3=128,
         mem="DDR5-5200", max_mem=512, litho=5, ecc=False, pcie_ver=5, pcie_lanes=48,
         aliases=["7960x", "tr 7960x", "threadripper 7960x"]),

    # ═══════════════════════════════════════════════════════════
    # AMD Threadripper PRO 7000 — Castle Peak (sWRX8, DDR5-5200)
    # No integrated GPU
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen Threadripper PRO 7995WX", "TR PRO 7995WX", CpuSocket.sWRX8,
         64, 128, 2.5, 5.1, 350, arch="Zen 4 Storm Peak", l3=256,
         mem="DDR5-5200", max_mem=2048, litho=5, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["7995wx", "tr pro 7995wx", "threadripper pro 7995wx"]),
    _amd("AMD Ryzen Threadripper PRO 7975WX", "TR PRO 7975WX", CpuSocket.sWRX8,
         32, 64, 4.0, 5.3, 350, arch="Zen 4 Storm Peak", l3=128,
         mem="DDR5-5200", max_mem=2048, litho=5, ecc=True, pcie_ver=5, pcie_lanes=128,
         aliases=["7975wx", "tr pro 7975wx", "threadripper pro 7975wx"]),

    # ═══════════════════════════════════════════════════════════
    # AMD Ryzen AI 300 Series — Strix Point (AM5, DDR5-5600)
    # iGPU: Radeon 890M (RDNA 3.5, 16CU) / 880M (RDNA 3.5, 12CU)
    # NPU: XDNA 2 (50 TOPS)
    # ═══════════════════════════════════════════════════════════
    _amd("AMD Ryzen AI 9 HX 370", "Ryzen AI 9 HX 370", CpuSocket.AM5,
         12, 24, 2.0, 5.1, 28, arch="Zen 5 Strix Point", l3=24,
         mem="DDR5-5600/LPDDR5X-7500", max_mem=256, litho=4, pcie_ver=4, pcie_lanes=24,
         aliases=["hx 370", "ryzen ai 9 hx 370", "ai 9 hx 370"],
         p_cores=4, e_cores=8, pl1=28, pl2=80,
         igpu_model="Radeon 890M", igpu_cus=16, igpu_clock=2900,
         igpu_pci="0x1507", igpu_tflops=11.9, igpu_av1_decode=True, igpu_av1_encode=True),
    _amd("AMD Ryzen AI 9 365", "Ryzen AI 9 365", CpuSocket.AM5,
         10, 20, 2.0, 5.0, 28, arch="Zen 5 Strix Point", l3=24,
         mem="DDR5-5600/LPDDR5X-7500", max_mem=256, litho=4, pcie_ver=4, pcie_lanes=24,
         aliases=["ai 9 365", "ryzen ai 9 365"],
         p_cores=4, e_cores=6, pl1=28, pl2=80,
         igpu_model="Radeon 880M", igpu_cus=12, igpu_clock=2900,
         igpu_pci="0x1507", igpu_tflops=8.9, igpu_av1_decode=True, igpu_av1_encode=True),

    # ═══════════════════════════════════════════════════════════
    # AMD EPYC 4004 Series — non-P / PX variants (AM5, DDR5-5200)
    # Additional: EPYC 4584PX (3D V-Cache), EPYC 4344P, EPYC 4144P
    # ═══════════════════════════════════════════════════════════
    _amd("AMD EPYC 4584PX", "EPYC 4584PX", CpuSocket.AM5,
         16, 32, 4.2, 5.2, 170, arch="Zen 4", l3=128, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4584px", "4584px"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD EPYC 4344P", "EPYC 4344P", CpuSocket.AM5,
         8, 16, 3.8, 5.0, 85, arch="Zen 4", l3=32, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4344p", "4344p"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),
    _amd("AMD EPYC 4144P", "EPYC 4144P", CpuSocket.AM5,
         4, 8, 3.9, 5.0, 65, arch="Zen 4", l3=16, mem="DDR5-5200", max_mem=256,
         litho=5, ecc=True, aliases=["epyc 4144p", "4144p"], pcie_ver=5, pcie_lanes=28,
         igpu_model="Radeon Graphics (RDNA 2)", igpu_cus=2, igpu_clock=2200,
         igpu_pci="0x1640", igpu_tflops=0.563),

    # ═══════════════════════════════════════════════════════════
    # Intel Atom C Series — Denverton (BGA, DDR4-2400)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Atom C3950", "Atom C3950", CpuSocket.BGA,
           16, 16, 1.7, 2.2, 65, arch="Denverton", l3=16,
           mem="DDR4-2400", max_mem=256, litho=14, ecc=True, pcie_ver=3, pcie_lanes=20,
           aliases=["c3950", "atom c3950"]),
    _intel("Intel Atom C3850", "Atom C3850", CpuSocket.BGA,
           8, 8, 1.8, 2.2, 50, arch="Denverton", l3=8,
           mem="DDR4-2400", max_mem=256, litho=14, ecc=True, pcie_ver=3, pcie_lanes=16,
           aliases=["c3850", "atom c3850"]),

    # ═══════════════════════════════════════════════════════════
    # Intel Atom P Series — Alder Lake-N (BGA, DDR4-3200/DDR5-4800)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Atom x7425E", "Atom x7425E", CpuSocket.BGA,
           4, 4, 1.5, 3.4, 12, arch="Alder Lake-N", l3=6,
           mem="DDR4-3200/DDR5-4800", max_mem=16, litho=10, ecc=False, pcie_ver=3, pcie_lanes=9,
           aliases=["x7425e", "atom x7425e"]),
    _intel("Intel Atom x7211E", "Atom x7211E", CpuSocket.BGA,
           2, 2, 1.0, 3.2, 10, arch="Alder Lake-N", l3=6,
           mem="DDR4-3200/DDR5-4800", max_mem=16, litho=10, ecc=False, pcie_ver=3, pcie_lanes=9,
           aliases=["x7211e", "atom x7211e"]),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon E-2100 Series — Coffee Lake (LGA1151, DDR4-2666)
    # iGPU: UHD P630/P630 (for G models)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon E-2186G", "Xeon E-2186G", CpuSocket.LGA1151,
           6, 12, 3.8, 4.7, 95, arch="Coffee Lake", l3=12,
           mem="DDR4-2666", max_mem=64, litho=14, ecc=True, pcie_ver=3, pcie_lanes=16,
           aliases=["e-2186g", "xeon e-2186g"],
           igpu_model="UHD P630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x3E9B", igpu_tflops=0.461),
    _intel("Intel Xeon E-2176G", "Xeon E-2176G", CpuSocket.LGA1151,
           6, 12, 3.7, 4.7, 95, arch="Coffee Lake", l3=12,
           mem="DDR4-2666", max_mem=64, litho=14, ecc=True, pcie_ver=3, pcie_lanes=16,
           aliases=["e-2176g", "xeon e-2176g"],
           igpu_model="UHD P630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x3E9B", igpu_tflops=0.461),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon E-2200 Series — Coffee Lake Refresh (LGA1151, DDR4-2666)
    # iGPU: UHD P630
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon E-2288G", "Xeon E-2288G", CpuSocket.LGA1151,
           8, 16, 3.7, 5.0, 95, arch="Coffee Lake R", l3=16,
           mem="DDR4-2666", max_mem=128, litho=14, ecc=True, pcie_ver=3, pcie_lanes=16,
           aliases=["e-2288g", "xeon e-2288g"],
           igpu_model="UHD P630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x3E9B", igpu_tflops=0.461),
    _intel("Intel Xeon E-2278G", "Xeon E-2278G", CpuSocket.LGA1151,
           8, 16, 3.4, 5.0, 95, arch="Coffee Lake R", l3=16,
           mem="DDR4-2666", max_mem=128, litho=14, ecc=True, pcie_ver=3, pcie_lanes=16,
           aliases=["e-2278g", "xeon e-2278g"],
           igpu_model="UHD P630", igpu_eus=24, igpu_clock=1200,
           igpu_pci="0x3E9B", igpu_tflops=0.461),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon D-2100 Series — Skylake-DE (BGA, DDR4-2400)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon D-2187NT", "Xeon D-2187NT", CpuSocket.BGA,
           16, 32, 2.0, 3.0, 110, arch="Skylake-DE", l3=22,
           mem="DDR4-2400", max_mem=512, litho=14, ecc=True, pcie_ver=3, pcie_lanes=24,
           aliases=["d-2187nt", "xeon d-2187nt"]),
    _intel("Intel Xeon D-2142IT", "Xeon D-2142IT", CpuSocket.BGA,
           8, 16, 1.9, 3.0, 65, arch="Skylake-DE", l3=11,
           mem="DDR4-2400", max_mem=512, litho=14, ecc=True, pcie_ver=3, pcie_lanes=24,
           aliases=["d-2142it", "xeon d-2142it"]),

    # ═══════════════════════════════════════════════════════════
    # Intel Xeon E5-2600 v4 Series — Broadwell-EP (LGA2011-3, DDR4-2400)
    # ═══════════════════════════════════════════════════════════
    _intel("Intel Xeon E5-2699 v4", "Xeon E5-2699 v4", CpuSocket.LGA2011_3,
           22, 44, 2.2, 3.6, 145, arch="Broadwell-EP", l3=55,
           mem="DDR4-2400", max_mem=1536, litho=14, ecc=True, pcie_ver=3, pcie_lanes=40,
           aliases=["e5-2699 v4", "e5 2699 v4", "xeon e5-2699 v4"]),
    _intel("Intel Xeon E5-2680 v4", "Xeon E5-2680 v4", CpuSocket.LGA2011_3,
           14, 28, 2.4, 3.3, 120, arch="Broadwell-EP", l3=35,
           mem="DDR4-2400", max_mem=1536, litho=14, ecc=True, pcie_ver=3, pcie_lanes=40,
           aliases=["e5-2680 v4", "e5 2680 v4", "xeon e5-2680 v4"]),

    # ═══════════════════════════════════════════════════════════
    # Ampere Altra — ARM Server (ARM socket, DDR4-3200)
    # ═══════════════════════════════════════════════════════════
    _arm("Ampere Altra Q80-30", "Altra Q80-30", 80, 80, 2.0, 3.3, 210,
         arch="Neoverse N1", l3=32, mem="DDR4-3200", max_mem=2048,
         litho=7, ecc=True, aliases=["q80-30", "altra q80-30"]),
    _arm("Ampere Altra Max M128-30", "Altra Max M128-30", 128, 128, 2.0, 3.3, 250,
         arch="Neoverse N1", l3=32, mem="DDR4-3200", max_mem=2048,
         litho=7, ecc=True, aliases=["m128-30", "altra max m128-30"]),
    _arm("AmpereOne A192-30", "AmpereOne A192-30", 192, 192, 2.0, 3.0, 350,
         arch="AmpereOne", l3=64, mem="DDR5-4800", max_mem=2048,
         litho=5, ecc=True, aliases=["a192-30", "ampereone a192-30"]),

    # ═══════════════════════════════════════════════════════════
    # AWS Graviton2 — ARM Server (DDR4-3200)
    # ═══════════════════════════════════════════════════════════
    _arm("AWS Graviton2", "Graviton2", 64, 64, 2.5, 3.0, 180,
         arch="Neoverse N1", l3=32, mem="DDR4-3200", max_mem=1024,
         litho=7, ecc=True, aliases=["graviton2", "aws graviton2"]),

    # ═══════════════════════════════════════════════════════════
    # AWS Graviton3 — ARM Server (DDR5-4800)
    # ═══════════════════════════════════════════════════════════
    _arm("AWS Graviton3", "Graviton3", 64, 64, 2.6, 3.0, 200,
         arch="Neoverse V1", l3=32, mem="DDR5-4800", max_mem=1024,
         litho=5, ecc=True, aliases=["graviton3", "aws graviton3"]),

    # ═══════════════════════════════════════════════════════════
    # AWS Graviton4 — ARM Server (DDR5-5600)
    # ═══════════════════════════════════════════════════════════
    _arm("AWS Graviton4", "Graviton4", 96, 96, 2.8, 3.2, 250,
         arch="Neoverse V2", l3=48, mem="DDR5-5600", max_mem=1536,
         litho=4, ecc=True, aliases=["graviton4", "aws graviton4"]),
]


class CpuCatalog:
    def __init__(self) -> None:
        self._entries = CPU_CATALOG
        self._by_short: dict[str, CpuCatalogEntry] = {}
        self._by_vendor: dict[str, list[CpuCatalogEntry]] = {}
        self._by_igpu_pci: dict[str, CpuCatalogEntry] = {}
        for e in self._entries:
            self._by_short[e.model_short.lower()] = e
            v = e.vendor.value
            if v not in self._by_vendor:
                self._by_vendor[v] = []
            self._by_vendor[v].append(e)
            if e.igpu and e.igpu.pci_device_id:
                pid = e.igpu.pci_device_id.lower().replace("0x", "")
                if pid not in self._by_igpu_pci:
                    self._by_igpu_pci[pid] = e

    @property
    def entries(self) -> list[CpuCatalogEntry]:
        return self._entries

    def lookup(self, name: str) -> CpuCatalogEntry | None:
        n = name.strip()
        if n.lower() in self._by_short:
            return self._by_short[n.lower()]
        for entry in self._entries:
            if entry.matches(n):
                return entry
        n_clean = n.lower().replace("intel ", "").replace("amd ", "")
        n_clean = n_clean.replace("-", " ").replace("_", " ")
        words = n_clean.split()
        best_score = 0.0
        best_entry: CpuCatalogEntry | None = None
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

    def lookup_by_igpu_pci(self, pci_id: str) -> CpuCatalogEntry | None:
        pid = pci_id.lower().strip()
        if pid.startswith("0x"):
            pid = pid[2:]
        return self._by_igpu_pci.get(pid)

    def query(self, vendor: str | None = None, socket: str | None = None,
              min_cores: int | None = None, has_igpu: bool | None = None,
              min_igpu_tflops: float | None = None) -> list[dict]:
        results = self._entries
        if vendor:
            results = [e for e in results if e.vendor.value == vendor.lower()]
        if socket:
            s = socket.lower()
            results = [e for e in results if (
                e.socket.value.lower() if isinstance(e.socket, CpuSocket) else str(e.socket).lower()
            ) == s]
        if min_cores:
            results = [e for e in results if e.cores >= min_cores]
        if has_igpu is not None:
            results = [e for e in results if (e.igpu is not None) == has_igpu]
        if min_igpu_tflops:
            results = [e for e in results if e.igpu and e.igpu.shader_tflops_fp32 >= min_igpu_tflops]
        return [e.to_dict() for e in results]

    def group_by_vendor(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for v, entries in self._by_vendor.items():
            result[v] = [e.to_dict() for e in entries]
        return result

    def group_by_socket(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for e in self._entries:
            s = e.socket.value if isinstance(e.socket, CpuSocket) else str(e.socket)
            if s not in result:
                result[s] = []
            result[s].append(e.to_dict())
        return result


_cpu_catalog_instance: CpuCatalog | None = None


def get_cpu_catalog() -> CpuCatalog:
    global _cpu_catalog_instance
    if _cpu_catalog_instance is None:
        _cpu_catalog_instance = CpuCatalog()
    return _cpu_catalog_instance


def lookup_cpu(name: str) -> CpuCatalogEntry | None:
    return get_cpu_catalog().lookup(name)
