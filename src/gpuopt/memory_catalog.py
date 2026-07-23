from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(Enum):
    DDR1 = "DDR1"
    DDR2 = "DDR2"
    DDR3 = "DDR3"
    DDR4 = "DDR4"
    DDR5 = "DDR5"
    LPDDR4 = "LPDDR4"
    LPDDR5 = "LPDDR5"
    LPDDR5X = "LPDDR5X"
    HBM2 = "HBM2"
    HBM2E = "HBM2E"
    HBM3 = "HBM3"
    HBM3E = "HBM3E"
    GDDR6 = "GDDR6"
    GDDR6X = "GDDR6X"
    GDDR7 = "GDDR7"
    UNKNOWN = "Unknown"

    @classmethod
    def from_smbios_type(cls, type_code: int) -> MemoryType:
        return _SMBIOS_TYPE_MAP.get(type_code, cls.UNKNOWN)

    @classmethod
    def from_name(cls, name: str) -> MemoryType:
        n = name.upper().strip()
        for mt in cls:
            if mt.value.upper() == n:
                return mt
        if "DDR5" in n:
            return cls.DDR5
        if "DDR4" in n:
            return cls.DDR4
        if "LPDDR5X" in n:
            return cls.LPDDR5X
        if "LPDDR5" in n:
            return cls.LPDDR5
        if "LPDDR4" in n:
            return cls.LPDDR4
        if "DDR3" in n:
            return cls.DDR3
        return cls.UNKNOWN


_SMBIOS_TYPE_MAP: dict[int, MemoryType] = {
    0x00: MemoryType.DDR1,
    0x11: MemoryType.DDR2,
    0x12: MemoryType.DDR3,
    0x13: MemoryType.DDR4,
    0x14: MemoryType.DDR5,     # SMBIOS spec 3.6+
    0x15: MemoryType.LPDDR5,
    0x16: MemoryType.LPDDR5X,
    0x1A: MemoryType.DDR4,     # Some implementations use 26 (0x1A)
    0x22: MemoryType.DDR5,     # 34 decimal
}


class FormFactor(Enum):
    DIMM = "DIMM"
    SODIMM = "SODIMM"
    RDIMM = "RDIMM"
    LRDIMM = "LRDIMM"
    UDIMM = "UDIMM"
    SO_RDIMM = "SO-RDIMM"
    SO_UDIMM = "SO-UDIMM"
    CAMM = "CAMM"
    CAMM2 = "CAMM2"
    UNKNOWN = "Unknown"

    @classmethod
    def from_smbios_code(cls, code: int) -> FormFactor:
        return _SMBIOS_FORM_FACTOR_MAP.get(code, cls.UNKNOWN)


_SMBIOS_FORM_FACTOR_MAP: dict[int, FormFactor] = {
    0x01: FormFactor.DIMM,       # 1 = Other (we map to DIMM)
    0x02: FormFactor.SODIMM,     # 2 = Unknown
    0x09: FormFactor.DIMM,       # 9 = DIMM
    0x0C: FormFactor.SODIMM,     # 12 = SODIMM
    0x0D: FormFactor.SODIMM,     # 13 = SORDIMM (Small Outline Registered)
    0x0E: FormFactor.SODIMM,     # 14 = SOUDIMM (Small Outline Unbuffered)
    0x0F: FormFactor.RDIMM,      # 15 = RDIMM
    0x10: FormFactor.UDIMM,      # 16 = UDIMM
    0x11: FormFactor.SO_RDIMM,   # 17 = SO-RDIMM
    0x12: FormFactor.LRDIMM,     # 18 = LRDIMM
    0x1B: FormFactor.CAMM2,      # 27 = CAMM2
}


class EccStatus(Enum):
    NONE = "none"
    ECC = "ecc"
    ECC_WITH_ON_DIE = "ecc_with_on_die"  # parity vs SECDED
    UNKNOWN = "unknown"


class MemoryCardinality(Enum):
    SINGLE_RANK = "single_rank"
    DUAL_RANK = "dual_rank"
    QUAD_RANK = "quad_rank"
    OCTAL_RANK = "octal_rank"
    UNKNOWN = "unknown"


RANK_MAP: dict[int, MemoryCardinality] = {
    1: MemoryCardinality.SINGLE_RANK,
    2: MemoryCardinality.DUAL_RANK,
    4: MemoryCardinality.QUAD_RANK,
    8: MemoryCardinality.OCTAL_RANK,
}


# ── JEDEC DDR4 Speed Bins ──
DDR4_SPEED_BINS: dict[str, int] = {
    "DDR4-1600": 1600,
    "DDR4-1866": 1866,
    "DDR4-2133": 2133,
    "DDR4-2400": 2400,
    "DDR4-2666": 2666,
    "DDR4-2933": 2933,
    "DDR4-3000": 3000,
    "DDR4-3200": 3200,
    "DDR4-3333": 3333,
    "DDR4-3466": 3466,
    "DDR4-3600": 3600,
    "DDR4-3733": 3733,
    "DDR4-3866": 3866,
    "DDR4-4000": 4000,
    "DDR4-4133": 4133,
    "DDR4-4266": 4266,
    "DDR4-4400": 4400,
    "DDR4-4533": 4533,
    "DDR4-4666": 4666,
    "DDR4-4800": 4800,
    "DDR4-5000": 5000,
    "DDR4-5066": 5066,
    "DDR4-5100": 5100,
    "DDR4-5200": 5200,
    "DDR4-5333": 5333,
}

# JEDEC DDR5 Speed Bins
DDR5_SPEED_BINS: dict[str, int] = {
    "DDR5-3200": 3200,
    "DDR5-3600": 3600,
    "DDR5-4000": 4000,
    "DDR5-4400": 4400,
    "DDR5-4800": 4800,
    "DDR5-5200": 5200,
    "DDR5-5600": 5600,
    "DDR5-6000": 6000,
    "DDR5-6200": 6200,
    "DDR5-6400": 6400,
    "DDR5-6600": 6600,
    "DDR5-6800": 6800,
    "DDR5-7000": 7000,
    "DDR5-7200": 7200,
    "DDR5-7400": 7400,
    "DDR5-7600": 7600,
    "DDR5-7800": 7800,
    "DDR5-8000": 8000,
    "DDR5-8200": 8200,
    "DDR5-8400": 8400,
    "DDR5-8600": 8600,
    "DDR5-8800": 8800,
    "DDR5-9000": 9000,
    "DDR5-9200": 9200,
    "DDR5-9400": 9400,
    "DDR5-9600": 9600,
}

ALL_DDR_BINS: dict[str, int] = {**DDR4_SPEED_BINS, **DDR5_SPEED_BINS}


def classify_speed(mhz: int, ddr_type: str | None = None) -> str:
    if ddr_type:
        bins = DDR5_SPEED_BINS if "5" in ddr_type else DDR4_SPEED_BINS
        closest = min(bins, key=lambda k: abs(bins[k] - mhz))
        return closest
    closest = min(ALL_DDR_BINS, key=lambda k: abs(ALL_DDR_BINS[k] - mhz))
    return closest


def memory_type_from_speed(mhz: int) -> MemoryType | None:
    if mhz < 3200:
        return MemoryType.DDR4
    if mhz < 4800:
        if mhz >= 4266:
            return None  # could be DDR4 OC or DDR5
        return MemoryType.DDR4
    if mhz >= 4800:
        return MemoryType.DDR5
    return None


# ── Bandwidth calculation utilities ──

def ddr_peak_bandwidth_gbps(mts: int, bus_bits: int = 64) -> float:
    if bus_bits <= 0 or mts <= 0:
        return 0.0
    bytes_per_transfer = bus_bits / 8
    return round(mts * bytes_per_transfer / 1000, 1)


# ── Manufacturer decoding ──

MANUFACTURER_LOOKUP: dict[str, str] = {
    "80AD": "Hynix",
    "80CE": "Samsung",
    "2C00": "Micron",
    "020F": "Hynix",
    "014F": "Transcend",
    "00AD": "Hynix",
    "00CE": "Samsung",
    "00FE": "Broadcom",
    "0100": "Smart Modular",
    "0108": "Infineon",
    "012C": "Corsair",
    "0142": "G.Skill",
    "014F": "Transcend Info",
    "0194": "Kingston",
    "01B4": "Rambus",
    "020F": "Hynix (older)",
    "02FE": "Broadcom",
    "04CD": "Hynix",
    "802C": "Micron",
    "80AD": "Hynix",
    "80CE": "Samsung",
    "859B": "Kingston",
    "84FE": "Broadcom",
    "830B": "Nanya",
    "827E": "Elpida",
}


def decode_manufacturer(code: str | None, part_number: str | None = None) -> str:
    if code and code.strip() in MANUFACTURER_LOOKUP:
        return MANUFACTURER_LOOKUP[code.strip()]
    if part_number:
        pn = part_number.upper()
        if pn.startswith("MT") or pn.startswith("MTA"):
            return "Micron"
        if pn.startswith("M") and ("SAMSUNG" in pn or pn.startswith("M3") or pn.startswith("M4") or pn.startswith("M5")):
            return "Samsung"
        if "HMA" in pn or "HMC" in pn:
            return "Hynix"
        if "KVR" in pn or "KF" in pn:
            return "Kingston"
        if "CM" in pn or "CMK" in pn or "CMD" in pn:
            return "Corsair"
        if "F4" in pn or "F5" in pn:
            return "G.Skill"
    return "Unknown"


# ── Samsung Part Number Decoding ──

SAMSUNG_MODEL_MAP: dict[str, str] = {
    "M3": "DDR3 / DDR4",
    "M4": "DDR4",
    "M5": "DDR5",
    "M3A": "DDR3",
    "M4A": "DDR4",
    "M5A": "DDR5",
    "M391": "DDR4 RDIMM",
    "M393": "DDR4 RDIMM (server)",
    "M471": "DDR4 SODIMM",
    "M474": "DDR4 SODIMM (server)",
    "M378": "DDR4 UDIMM",
    "M425": "DDR5 RDIMM",
    "M426": "DDR5 SODIMM",
    "M425R": "DDR5 RDIMM",
    "M426R": "DDR5 SODIMM",
    "M425L": "DDR5 LRDIMM",
}


# ── Micron Part Number Decoding ──

MICRON_MODEL_MAP: dict[str, str] = {
    "MT40": "DDR4",
    "MT60": "DDR5",
    "MTC10": "DDR5",
    "MTC20": "DDR5 (server)",
    "MT8": "DDR3",
    "MT18": "DDR4",
    "MTA18": "DDR4 (server)",
}


# ── Hynix Part Number Decoding ──

HYNIX_MODEL_MAP: dict[str, str] = {
    "HMA": "DDR4",
    "HMC": "DDR5",
    "HMT": "DDR3",
    "H5T": "DDR3/DDR4",
    "H58": "DDR5",
    "H9H": "HBM",
}


def decode_part_number(pn: str) -> dict[str, Any]:
    pn = pn.strip().upper()
    info: dict[str, Any] = {"raw": pn}
    if pn.startswith("MT") or pn.startswith("MTC"):
        for prefix, desc in MICRON_MODEL_MAP.items():
            if pn.startswith(prefix):
                info["type"] = desc
                info["vendor"] = "Micron"
                break
    elif pn.startswith("M"):
        for prefix, desc in SAMSUNG_MODEL_MAP.items():
            if pn.startswith(prefix):
                info["type"] = desc
                info["vendor"] = "Samsung"
                break
    elif pn.startswith("H"):
        for prefix, desc in HYNIX_MODEL_MAP.items():
            if pn.startswith(prefix):
                info["type"] = desc
                info["vendor"] = "Hynix"
                break
    info.setdefault("vendor", "Unknown")
    return info


# ── ECC Detection ──

def detect_ecc(total_width_bits: int, data_width_bits: int) -> EccStatus:
    if total_width_bits <= 0 or data_width_bits <= 0:
        return EccStatus.UNKNOWN
    extra = total_width_bits - data_width_bits
    if extra == 8 and data_width_bits == 64:
        return EccStatus.ECC
    if extra == 0:
        return EccStatus.NONE
    if extra > 0:
        return EccStatus.ECC
    return EccStatus.UNKNOWN


def detect_registered(form_factor: str | FormFactor | None,
                      smbios_type_detail: str | None = None) -> str | None:
    if isinstance(form_factor, FormFactor):
        ff = form_factor.value.upper()
    elif form_factor:
        ff = form_factor.upper()
    else:
        ff = ""

    if "RDIMM" in ff or "LRDIMM" in ff or "SO-RDIMM" in ff:
        return "registered"
    if "UDIMM" in ff or "SO-UDIMM" in ff:
        return "unbuffered"
    if "DIMM" in ff and "REG" in ff:
        return "registered"
    if "DIMM" in ff:
        return "unbuffered"
    if "SODIMM" in ff:
        return "unbuffered"

    if smbios_type_detail:
        sd = smbios_type_detail.upper()
        if "REGISTERED" in sd:
            return "registered"
        if "UNBUFFERED" in sd:
            return "unbuffered"
        if "LOAD REDUCED" in sd:
            return "load_reduced"
    return None


# ── SMBIOS Memory Parsing ──

@dataclass
class SmbiosDimmInfo:
    locator: str = ""
    bank_locator: str = ""
    manufacturer: str = ""
    part_number: str = ""
    serial_number: str = ""
    size_mb: int = 0
    speed_mhz: int = 0
    configured_speed_mhz: int = 0
    memory_type: str = ""
    memory_type_code: int = 0
    form_factor_code: int = 0
    form_factor: FormFactor = FormFactor.UNKNOWN
    data_width_bits: int = 0
    total_width_bits: int = 0
    rank: int = 0
    voltage: float = 0.0
    manufacturer_id: str = ""
    cas_latency: str = ""
    tRCD: str = ""
    tRP: str = ""
    tRAS: str = ""
    min_tcycle: str = ""
    pmic_manufacturer: str = ""
    pmic_part_number: str = ""
    thermal_sensor: bool = False

    def classify_type(self) -> MemoryType:
        if self.memory_type_code:
            return MemoryType.from_smbios_type(self.memory_type_code)
        return MemoryType.from_name(self.memory_type)

    def classify_form_factor(self) -> FormFactor:
        return FormFactor.from_smbios_code(self.form_factor_code)

    def ecc_status(self) -> EccStatus:
        return detect_ecc(self.total_width_bits, self.data_width_bits)

    def is_ecc(self) -> bool:
        return self.ecc_status() in (EccStatus.ECC, EccStatus.ECC_WITH_ON_DIE)

    def cardinality(self) -> MemoryCardinality:
        return RANK_MAP.get(self.rank, MemoryCardinality.UNKNOWN)

    def speed_label(self) -> str:
        return classify_speed(self.speed_mhz, self.memory_type)

    def bandwidth_gbps(self) -> float:
        return ddr_peak_bandwidth_gbps(self.speed_mhz if self.speed_mhz else self.configured_speed_mhz)

    def to_dict(self) -> dict:
        return {
            "locator": self.locator,
            "bank_locator": self.bank_locator,
            "manufacturer": decode_manufacturer(self.manufacturer_id, self.part_number) if not self.manufacturer else self.manufacturer,
            "part_number": self.part_number,
            "serial_number": self.serial_number,
            "size_mb": self.size_mb,
            "size_gb": round(self.size_mb / 1024, 1) if self.size_mb else 0.0,
            "speed_mhz": self.speed_mhz,
            "configured_speed_mhz": self.configured_speed_mhz,
            "speed_label": self.speed_label(),
            "memory_type": self.classify_type().value,
            "form_factor": self.classify_form_factor().value,
            "data_width_bits": self.data_width_bits,
            "total_width_bits": self.total_width_bits,
            "ecc": self.is_ecc(),
            "ecc_status": self.ecc_status().value,
            "rank": self.rank,
            "cardinality": self.cardinality().value,
            "voltage": self.voltage,
            "bandwidth_gbps": self.bandwidth_gbps(),
            "cas_latency": self.cas_latency,
            "tRCD": self.tRCD,
            "tRP": self.tRP,
            "tRAS": self.tRAS,
            "min_tcycle": self.min_tcycle,
            "pmic_manufacturer": self.pmic_manufacturer,
            "pmic_part_number": self.pmic_part_number,
            "thermal_sensor": self.thermal_sensor,
        }


# ── DIMM Capacity Lookup ──

# ── /sys-based memory detection (container-friendly) ──

def detect_memory_from_sysfs() -> SystemMemorySummary:
    import os
    total_gb = 0.0
    try:
        mem_total_path = "/sys/devices/system/memory"
        if not os.path.isdir(mem_total_path):
            return _detect_from_cgroup()
        online_blocks = 0
        for entry in os.listdir(mem_total_path):
            if entry.startswith("memory"):
                state_path = os.path.join(mem_total_path, entry, "state")
                if os.path.isfile(state_path):
                    with open(state_path) as f:
                        if f.read().strip() == "online":
                            online_blocks += 1
        block_size_path = "/sys/devices/system/memory/block_size_bytes"
        if os.path.isfile(block_size_path):
            with open(block_size_path) as f:
                block_size = int(f.read().strip(), 16)
                total_gb = round(online_blocks * block_size / (1024**3), 1)
    except (OSError, FileNotFoundError, ValueError):
        pass
    if total_gb == 0:
        return _detect_from_cgroup()
    return SystemMemorySummary(
        total_size_gb=total_gb,
        memory_type=MemoryType.UNKNOWN,
        form_factor=FormFactor.UNKNOWN,
        ecc_enabled=False,
        num_dimms=0,
        memory_type_detail="detected via /sys (container)",
    )


def _detect_from_cgroup() -> SystemMemorySummary:
    import os
    try:
        mem_paths = [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]
        for path in mem_paths:
            if os.path.isfile(path):
                with open(path) as f:
                    val = f.read().strip()
                if val and val != "max":
                    total_gb = round(int(val) / (1024**3), 1)
                    return SystemMemorySummary(
                        total_size_gb=total_gb,
                        memory_type=MemoryType.UNKNOWN,
                        form_factor=FormFactor.UNKNOWN,
                        ecc_enabled=False,
                        num_dimms=0,
                        memory_type_detail="detected via cgroup (container)",
                    )
    except (OSError, FileNotFoundError, ValueError):
        pass
    return SystemMemorySummary()


DIMM_CAPACITIES_GB: dict[str, list[int]] = {
    "DDR4_DIMM": [4, 8, 16, 32, 64, 128, 256],
    "DDR4_SODIMM": [4, 8, 16, 32, 64],
    "DDR4_RDIMM": [8, 16, 32, 64, 128, 256, 512],
    "DDR4_LRDIMM": [64, 128, 256],
    "DDR5_DIMM": [8, 16, 24, 32, 48, 64, 96, 128],
    "DDR5_SODIMM": [8, 16, 24, 32, 48, 64],
    "DDR5_RDIMM": [16, 32, 48, 64, 96, 128, 256, 512],
    "DDR5_LRDIMM": [128, 256, 512],
    "DDR5_CAMM2": [8, 16, 32, 64, 128, 256],
}


# ── Aggregate System Memory ──

@dataclass
class SystemMemorySummary:
    dimms: list[SmbiosDimmInfo] = field(default_factory=list)
    total_size_gb: float = 0.0
    memory_type: MemoryType = MemoryType.UNKNOWN
    form_factor: FormFactor = FormFactor.UNKNOWN
    max_speed_mhz: int = 0
    min_speed_mhz: int = 0
    config_speed_mhz: int = 0
    ecc_enabled: bool = False
    num_dimms: int = 0
    num_channels: int = 0
    channel_info: str = ""
    memory_type_detail: str = ""

    def to_dict(self) -> dict:
        return {
            "total_size_gb": self.total_size_gb,
            "memory_type": self.memory_type.value,
            "form_factor": self.form_factor.value,
            "max_speed_mhz": self.max_speed_mhz,
            "min_speed_mhz": self.min_speed_mhz,
            "configured_speed_mhz": self.config_speed_mhz,
            "ecc_enabled": self.ecc_enabled,
            "num_dimms": self.num_dimms,
            "num_channels": self.num_channels,
            "channel_info": self.channel_info,
            "memory_type_detail": self.memory_type_detail,
            "dimms": [d.to_dict() for d in self.dimms],
        }

    @classmethod
    def from_dimm_list(cls, dimms: list[SmbiosDimmInfo]) -> SystemMemorySummary:
        if not dimms:
            return cls()
        total_mb = sum(d.size_mb for d in dimms if d.size_mb)
        speeds = [d.speed_mhz for d in dimms if d.speed_mhz]
        config_speeds = [d.configured_speed_mhz for d in dimms if d.configured_speed_mhz]
        types = {d.classify_type() for d in dimms}
        forms = {d.classify_form_factor() for d in dimms}
        eccs = [d.is_ecc() for d in dimms]

        memory_type = MemoryType.UNKNOWN
        for t in (MemoryType.DDR5, MemoryType.DDR4, MemoryType.LPDDR5, MemoryType.LPDDR4):
            if t in types:
                memory_type = t
                break
        form_factor = FormFactor.UNKNOWN
        for f in (FormFactor.DIMM, FormFactor.SODIMM, FormFactor.RDIMM,
                  FormFactor.LRDIMM, FormFactor.CAMM2):
            if f in forms:
                form_factor = f
                break

        locators = [d.locator for d in dimms if d.locator]
        num_channels = len({d.bank_locator for d in dimms if d.bank_locator})

        return cls(
            dimms=list(dimms),
            total_size_gb=round(total_mb / 1024, 1),
            memory_type=memory_type,
            form_factor=form_factor,
            max_speed_mhz=max(speeds) if speeds else 0,
            min_speed_mhz=min(speeds) if speeds else 0,
            config_speed_mhz=max(config_speeds) if config_speeds else 0,
            ecc_enabled=any(eccs) if eccs else False,
            num_dimms=len(dimms),
            num_channels=max(num_channels, 1),
            channel_info=f"{len(dimms)}x DIMMs in {max(num_channels, 1)} channels" if dimms else "",
            memory_type_detail=f"{memory_type.value} @ {max(speeds) if speeds else 0} MHz",
        )


# ── SPD CAS Latency Extraction ──

_SPD_LINE_RE = _re.compile(r'^([0-9a-fA-F]+):\s+(.+)')


def _parse_spd_line(line: str) -> tuple[int, dict[int, int]] | None:
    """Parse a single line of hex dump (i2cdump or xxd style). Returns (base, {offset: value})."""
    m = _SPD_LINE_RE.match(line)
    if not m:
        return None
    try:
        base = int(m.group(1), 16)
    except ValueError:
        return None
    raw = m.group(2)
    hex_part = ""
    for ch in raw:
        if ch in "0123456789abcdefABCDEF ":
            hex_part += ch
        else:
            break
    hex_part = hex_part.strip()
    tokens = hex_part.split()
    if not tokens:
        return None
    values: dict[int, int] = {}
    for i, tok in enumerate(tokens):
        try:
            val = int(tok, 16)
        except ValueError:
            continue
        if len(tok) == 2:
            values[base + i] = val
        elif len(tok) == 4:
            values[base + i * 2] = (val >> 8) & 0xFF
            values[base + i * 2 + 1] = val & 0xFF
    return base, values


def parse_spd_hex_dump(text: str) -> dict[int, int]:
    """Parse i2cdump (hexdump -C style) or xxd output into byte offset -> value map."""
    result: dict[int, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _parse_spd_line(line)
        if parsed:
            _, values = parsed
            result.update(values)
    return result


def extract_spd_cas_latency_byte(hex_dump: str, byte_offset: int = 18) -> int | None:
    """Extract the raw byte at *byte_offset* from an SPD hex dump (e.g. byte 18 for DDR4 CL)."""
    parsed = parse_spd_hex_dump(hex_dump)
    return parsed.get(byte_offset)


def extract_spd_cas_latency_from_sysfs(eeprom_path: str | None = None) -> dict[str, Any]:
    """Read an SPD EEPROM via sysfs and return raw timing bytes.

    Typical paths: /sys/bus/i2c/devices/0-0050/eeprom or /sys/bus/i2c/drivers/ee1004/*/eeprom.
    """
    import os
    paths: list[str] = []
    if eeprom_path:
        if os.path.isfile(eeprom_path):
            paths = [eeprom_path]
    else:
        for root in ("/sys/bus/i2c/devices", "/sys/bus/i2c/drivers/ee1004"):
            if os.path.isdir(root):
                for ent in os.listdir(root):
                    candidate = os.path.join(root, ent, "eeprom")
                    if os.path.isfile(candidate) and os.access(candidate, os.R_OK):
                        paths.append(candidate)
    if not paths:
        return {"error": "no SPD EEPROM found", "paths_checked": paths}
    try:
        with open(paths[0], "rb") as f:
            raw = f.read(512)
    except (OSError, PermissionError) as exc:
        return {"error": str(exc), "path": paths[0]}
    hex_str = " ".join(f"{b:02X}" for i, b in enumerate(raw))
    return extract_spd_cas_latency_from_raw(raw_bytes=raw, speed_mts=0)


def extract_spd_cas_latency_from_raw(raw_bytes: bytes, speed_mts: int = 0) -> dict[str, Any]:
    """Decode CAS latency from raw SPD bytes (JEDEC DDR4/DDR5).

    DDR4: byte 18 = CL in cycles.
    DDR5: bytes 18-19 = tCKAVGmin fine offset (1/256 of MTB = 0.125 ns).
    """
    if len(raw_bytes) < 20:
        return {"error": "SPD data too short"}
    key_byte = raw_bytes[0x0A] if len(raw_bytes) > 0x0A else 0
    is_ddr5 = key_byte == 0x05
    byte_18 = raw_bytes[18]
    byte_19 = raw_bytes[19] if len(raw_bytes) > 19 else 0

    result: dict[str, Any] = {
        "spd_revision": f"{raw_bytes[1] >> 4}.{raw_bytes[1] & 0x0F}" if len(raw_bytes) > 1 else "unknown",
        "dram_type": "DDR5" if is_ddr5 else "DDR4",
        "byte_18_hex": f"0x{byte_18:02X}",
        "byte_19_hex": f"0x{byte_19:02X}",
    }
    if is_ddr5 and speed_mts > 0:
        # DDR5: MTB = 0.125 ns for timing parameters
        # tCKAVGmin is stored at byte 18 (MTB value, units of 0.125 ns)
        tck_ps = byte_18 * 0.125 * 1000  # picoseconds
        tck_from_speed_ps = 2000000.0 / speed_mts  # tCK = 2 / speed * 1e6 ps
        if tck_ps > 0:
            cl = round(tck_ps / tck_from_speed_ps, 1)
        else:
            cl = round(speed_mts / 2000 * 0.125, 1)  # rough estimate
        result["cas_latency_cycles"] = cl
        result["tck_ps"] = round(tck_ps, 1)
        result["tck_from_speed_ps"] = round(tck_from_speed_ps, 1)
    elif not is_ddr5:
        # DDR4: byte 18 = CL
        result["cas_latency_cycles"] = byte_18
    if len(raw_bytes) > 0x17:
        result["tRCD_byte"] = f"0x{raw_bytes[0x17]:02X}"
    if len(raw_bytes) > 0x19:
        result["tRP_byte"] = f"0x{raw_bytes[0x19]:02X}"
    if len(raw_bytes) > 0x1B:
        result["tRAS_byte"] = f"0x{raw_bytes[0x1B]:02X}"
    return result


def extract_spd_cas_latency_from_decode_dimms(text: str) -> dict[str, Any]:
    """Parse the output of ``decode-dimms`` to extract CAS latency."""
    result: dict[str, Any] = {}
    patterns = {
        "cas_latency": _re.compile(r'CAS\s+Latency\s*\(CL\)\s*:\s*(\d+(?:\.\d+)?)', _re.IGNORECASE),
        "tRCD": _re.compile(r'tRCD\s*:\s*(\d+(?:\.\d+)?)', _re.IGNORECASE),
        "tRP": _re.compile(r'tRP\s*:\s*(\d+(?:\.\d+)?)', _re.IGNORECASE),
        "tRAS": _re.compile(r'tRAS\s*:\s*(\d+(?:\.\d+)?)', _re.IGNORECASE),
        "speed": _re.compile(r'Speed\s*:\s*(\d+(?:\.\d+)?)\s*MHz', _re.IGNORECASE),
        "memory_type": _re.compile(r'Memory\s+Type\s*:\s*(\S+)', _re.IGNORECASE),
    }
    for key, pat in patterns.items():
        m = pat.search(text)
        if m:
            try:
                result[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            except ValueError:
                result[key] = m.group(1)
    return result
