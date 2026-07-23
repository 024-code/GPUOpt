from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import get_settings
from .cpu_catalog import get_cpu_catalog
from .gpu_catalog import get_gpu_catalog
from .ml.web_datasets import WebDatasetIngestion
from .repository import ClusterRepository
from .schemas import ClusterCreate
from .services import EnvironmentCheckService


def _load_environments(path: Path) -> list[ClusterCreate]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    clusters = data.get("clusters", [])
    return [ClusterCreate.model_validate(cluster) for cluster in clusters]


def command_seed(args: argparse.Namespace) -> int:
    repository = ClusterRepository(get_settings().database_path)
    for payload in _load_environments(Path(args.file)):
        record = repository.upsert_cluster(payload)
        print(f"registered {record.environment}/{record.name} ({record.id})")
    return 0


def command_check_all(args: argparse.Namespace) -> int:
    repository = ClusterRepository(get_settings().database_path)
    service = EnvironmentCheckService(repository)
    if args.file:
        for payload in _load_environments(Path(args.file)):
            repository.upsert_cluster(payload)
    reports = service.check_all()
    output = [report.model_dump(mode="json") for report in reports]
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for report in reports:
            print(f"{report.environment:12} {report.cluster_name:24} {report.overall_status.value.upper()}")
            for check in report.checks:
                print(f"  {check.status.value:5} {check.name:20} {check.message}")
    return 1 if any(report.overall_status.value == "fail" for report in reports) else 0


def command_dataset_list(args: argparse.Namespace) -> int:
    ingestion = WebDatasetIngestion()
    datasets = ingestion.list_datasets()
    print(f"{'Name':35} {'Cached':8} {'Telemetry':10} {'Samples':8}")
    print("-" * 65)
    for ds in datasets:
        cached = "yes" if ds.get("cached") else "no"
        samples = ds.get("size_bytes", "—")
        print(f"{ds.get('name', ds.get('local_file', '?')):35} {cached:8} {str(ds.get('telemetry', '—')):10} {samples}")
    if not datasets:
        print("No datasets found.")
    return 0


def command_dataset_download(args: argparse.Namespace) -> int:
    ingestion = WebDatasetIngestion()
    if args.name == "all":
        results = ingestion.ingest_all_available()
        for name, info in results.items():
            if name.startswith("_"):
                continue
            status = "OK" if info.get("samples", 0) > 0 else "FAIL"
            print(f"  {status:6} {name:35} {info.get('samples', 0)} samples")
    else:
        path = ingestion.download_dataset(args.name, force=args.force)
        data = ingestion.ingest(args.name)
        print(f"Downloaded {args.name} to {path}")
        print(f"  Samples: {len(data)}")
    return 0


def command_gpu_list(args: argparse.Namespace) -> int:
    catalog = get_gpu_catalog()
    entries = catalog.query(
        vendor=args.vendor, segment=args.segment,
        min_vram=args.min_vram,
        capabilities=[c.strip() for c in args.capabilities.split(",")] if args.capabilities else None,
    )
    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        print(f"{'Model':42} {'VRAM':8} {'Bus':6} {'TDP':6} {'TF16':8} {'Capabilities'}")
        print("-" * 120)
        for e in entries:
            caps = ", ".join(e.get("capabilities", [])[:4])
            bw = e.get("memory_bus_bits", 0)
            print(f"{e['model_short']:42} {e['vram_gib']:>4.0f}GB {bw:>4}bit {e['tdp_watts']:>4.0f}W {e['tensor_tflops_fp16']:>6.0f}  {caps}")
        print(f"\nTotal: {len(entries)} GPUs")
    return 0


def command_gpu_lookup(args: argparse.Namespace) -> int:
    from .gpu_catalog import lookup_gpu
    entry = lookup_gpu(args.name)
    if entry:
        print(json.dumps(entry.to_dict(), indent=2))
    else:
        print(f"No GPU found matching '{args.name}'")
    return 1 if not entry else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuopt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="Register clusters from a YAML file")
    seed.add_argument("--file", required=True)
    seed.set_defaults(func=command_seed)

    check_all = subparsers.add_parser("check-all", help="Run checks against all registered clusters")
    check_all.add_argument("--file", help="Optionally upsert clusters before checking")
    check_all.add_argument("--json", action="store_true")
    check_all.set_defaults(func=command_check_all)

    dataset = subparsers.add_parser("dataset", help="Manage web datasets for training")
    dataset_sub = dataset.add_subparsers(dest="subcommand", required=True)
    ds_list = dataset_sub.add_parser("list", help="List available datasets")
    ds_list.set_defaults(func=command_dataset_list)
    ds_dl = dataset_sub.add_parser("download", help="Download a dataset")
    ds_dl.add_argument("name", help="Dataset name or 'all'")
    ds_dl.add_argument("--force", action="store_true", help="Force re-download")
    ds_dl.set_defaults(func=command_dataset_download)

    gpu = subparsers.add_parser("gpu", help="Query GPU catalog")
    gpu_sub = gpu.add_subparsers(dest="subcommand", required=True)
    gpu_list = gpu_sub.add_parser("list", help="List GPUs with optional filters")
    gpu_list.add_argument("--vendor", help="Filter by vendor (nvidia, amd, intel)")
    gpu_list.add_argument("--segment", help="Filter by segment (consumer, workstation, data_center, entry)")
    gpu_list.add_argument("--min-vram", type=float, help="Minimum VRAM in GB")
    gpu_list.add_argument("--capabilities", help="Comma-separated capability filters")
    gpu_list.add_argument("--json", action="store_true", help="JSON output")
    gpu_list.set_defaults(func=command_gpu_list)
    gpu_lookup = gpu_sub.add_parser("lookup", help="Look up a GPU by name")
    gpu_lookup.add_argument("name", help="GPU model name to look up")
    gpu_lookup.set_defaults(func=command_gpu_lookup)

    cpu = subparsers.add_parser("cpu", help="Query CPU catalog")
    cpu_sub = cpu.add_subparsers(dest="subcommand", required=True)
    cpu_list = cpu_sub.add_parser("list", help="List CPUs with optional filters")
    cpu_list.add_argument("--vendor", help="Filter by vendor (intel, amd)")
    cpu_list.add_argument("--socket", help="Filter by socket (lga1700, lga1200, am4, am5)")
    cpu_list.add_argument("--min-cores", type=int, help="Minimum core count")
    cpu_list.add_argument("--has-igpu", action="store_true", help="Only show CPUs with integrated GPU")
    cpu_list.add_argument("--min-igpu-tflops", type=float, help="Minimum iGPU TFLOPS")
    cpu_list.add_argument("--json", action="store_true", help="JSON output")
    cpu_list.set_defaults(func=command_cpu_list)
    cpu_lookup = cpu_sub.add_parser("lookup", help="Look up a CPU by name")
    cpu_lookup.add_argument("name", help="CPU model name to look up")
    cpu_lookup.set_defaults(func=command_cpu_lookup)

    ram = subparsers.add_parser("ram", help="Detect and query RAM")
    ram_sub = ram.add_subparsers(dest="subcommand", required=True)
    ram_detect = ram_sub.add_parser("detect", help="Detect local system memory via dmidecode")
    ram_detect.add_argument("--json", action="store_true", help="JSON output")
    ram_detect.set_defaults(func=command_ram_detect)

    hw = subparsers.add_parser("hardware", help="Detect local hardware")
    hw.set_defaults(func=command_hardware_detect)

    return parser


def command_cpu_list(args: argparse.Namespace) -> int:
    catalog = get_cpu_catalog()
    entries = catalog.query(
        vendor=args.vendor, socket=args.socket,
        min_cores=args.min_cores, has_igpu=args.has_igpu,
        min_igpu_tflops=args.min_igpu_tflops,
    )
    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        print(f"{'Model':42} {'C/T':8} {'Base':6} {'Boost':6} {'TDP':6} {'Socket':12} {'iGPU':30}")
        print("-" * 120)
        for e in entries:
            igpu = e.get("igpu", {})
            igpu_name = igpu.get("model", "—") if igpu else "—"
            print(f"{e['model_short']:42} {e['cores']}/{e['threads']:5} {e['base_clock_ghz']:>4.1f}G {e['boost_clock_ghz']:>4.1f}G {e['tdp_watts']:>4.0f}W {e['socket']:12} {igpu_name:30}")
        print(f"\nTotal: {len(entries)} CPUs")
    return 0


def command_cpu_lookup(args: argparse.Namespace) -> int:
    from .cpu_catalog import lookup_cpu
    entry = lookup_cpu(args.name)
    if entry:
        print(json.dumps(entry.to_dict(), indent=2))
    else:
        print(f"No CPU found matching '{args.name}'")
    return 1 if not entry else 0


def command_ram_detect(args: argparse.Namespace) -> int:
    from .hardware_detector import detect_local_memory
    mem = detect_local_memory()
    d = mem.to_dict()
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print(f"Total RAM: {d['total_size_gb']} GB")
        print(f"Type: {d['memory_type']}")
        print(f"Form Factor: {d['form_factor']}")
        print(f"Max Speed: {d['max_speed_mhz']} MHz")
        print(f"ECC: {d['ecc_enabled']}")
        print(f"DIMMs: {d['num_dimms']}")
        print(f"Channels: {d['num_channels']}")
        for di in d.get("dimms", []):
            print(f"  {di['locator'] or '?'}: {di['size_gb']} GB {di['speed_label']} "
                  f"{di['memory_type']} {di['form_factor']} "
                  f"{'ECC' if di['ecc'] else 'non-ECC'} Rank={di['rank']}")
    return 0


def command_hardware_detect(args: argparse.Namespace) -> int:
    from .hardware_detector import detect_local_hardware
    hw = detect_local_hardware()
    print(json.dumps(hw, indent=2))
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
