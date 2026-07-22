from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import get_settings
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
