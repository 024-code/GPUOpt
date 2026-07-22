from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

GPU_PRICING_CACHE: dict[str, list[dict[str, Any]]] = {}


@dataclass
class CloudPricingRow:
    provider: str
    region: str
    instance_type: str
    gpu_model: str
    gpu_count: int
    hourly_cost: float
    monthly_cost: float
    spot_hourly: float | None = None
    vcpu_count: int = 0
    memory_gb: float = 0.0
    source: str = "static"


AWS_GPU_INSTANCES: list[dict[str, Any]] = [
    {"instance": "p3.2xlarge", "gpu": "V100", "count": 1, "hourly": 3.06, "vcpu": 8, "mem": 61},
    {"instance": "p3.8xlarge", "gpu": "V100", "count": 4, "hourly": 12.24, "vcpu": 32, "mem": 244},
    {"instance": "p3.16xlarge", "gpu": "V100", "count": 8, "hourly": 24.48, "vcpu": 64, "mem": 488},
    {"instance": "p4d.24xlarge", "gpu": "A100", "count": 8, "hourly": 32.77, "vcpu": 96, "mem": 1152},
    {"instance": "p4de.24xlarge", "gpu": "A100", "count": 8, "hourly": 40.96, "vcpu": 96, "mem": 1152},
    {"instance": "p5.48xlarge", "gpu": "H100", "count": 8, "hourly": 49.36, "vcpu": 192, "mem": 2048},
    {"instance": "g4dn.xlarge", "gpu": "T4", "count": 1, "hourly": 0.526, "vcpu": 4, "mem": 16},
    {"instance": "g4dn.12xlarge", "gpu": "T4", "count": 4, "hourly": 3.912, "vcpu": 48, "mem": 192},
    {"instance": "g5.xlarge", "gpu": "A10G", "count": 1, "hourly": 1.006, "vcpu": 4, "mem": 16},
    {"instance": "g5.48xlarge", "gpu": "A10G", "count": 4, "hourly": 8.144, "vcpu": 192, "mem": 768},
    {"instance": "g6.xlarge", "gpu": "L4", "count": 1, "hourly": 1.044, "vcpu": 4, "mem": 16},
    {"instance": "g6.48xlarge", "gpu": "L4", "count": 4, "hourly": 8.352, "vcpu": 192, "mem": 768},
    {"instance": "trn1.32xlarge", "gpu": "Trainium", "count": 16, "hourly": 23.99, "vcpu": 128, "mem": 512},
]

AZURE_GPU_INSTANCES: list[dict[str, Any]] = [
    {"instance": "NC6s v3", "gpu": "V100", "count": 1, "hourly": 3.06, "vcpu": 6, "mem": 112},
    {"instance": "NC12s v3", "gpu": "V100", "count": 2, "hourly": 6.12, "vcpu": 12, "mem": 224},
    {"instance": "NC24s v3", "gpu": "V100", "count": 4, "hourly": 12.24, "vcpu": 24, "mem": 448},
    {"instance": "ND40s v2", "gpu": "V100", "count": 8, "hourly": 24.48, "vcpu": 40, "mem": 672},
    {"instance": "ND96asr v4", "gpu": "A100", "count": 4, "hourly": 18.37, "vcpu": 96, "mem": 900},
    {"instance": "ND96amsr v4", "gpu": "A100", "count": 8, "hourly": 36.74, "vcpu": 96, "mem": 1800},
    {"instance": "ND H100 v5", "gpu": "H100", "count": 8, "hourly": 56.23, "vcpu": 192, "mem": 2048},
    {"instance": "NC4as T4 v3", "gpu": "T4", "count": 1, "hourly": 0.75, "vcpu": 4, "mem": 28},
    {"instance": "NC8as T4 v3", "gpu": "T4", "count": 1, "hourly": 1.50, "vcpu": 8, "mem": 56},
]

GCP_GPU_INSTANCES: list[dict[str, Any]] = [
    {"instance": "a2-highgpu-1g", "gpu": "A100", "count": 1, "hourly": 3.67, "vcpu": 12, "mem": 85},
    {"instance": "a2-highgpu-4g", "gpu": "A100", "count": 4, "hourly": 14.68, "vcpu": 48, "mem": 340},
    {"instance": "a2-highgpu-8g", "gpu": "A100", "count": 8, "hourly": 29.36, "vcpu": 96, "mem": 680},
    {"instance": "a2-megagpu-16g", "gpu": "A100", "count": 16, "hourly": 58.72, "vcpu": 192, "mem": 1360},
    {"instance": "g2-standard-4", "gpu": "L4", "count": 1, "hourly": 0.97, "vcpu": 4, "mem": 16},
    {"instance": "g2-standard-8", "gpu": "L4", "count": 1, "hourly": 1.54, "vcpu": 8, "mem": 32},
    {"instance": "g2-standard-96", "gpu": "L4", "count": 4, "hourly": 7.76, "vcpu": 96, "mem": 384},
    {"instance": "n1-standard-4-t4", "gpu": "T4", "count": 1, "hourly": 0.76, "vcpu": 4, "mem": 15},
    {"instance": "n1-standard-8-t4", "gpu": "T4", "count": 2, "hourly": 1.52, "vcpu": 8, "mem": 30},
    {"instance": "n1-standard-16-h100", "gpu": "H100", "count": 8, "hourly": 52.80, "vcpu": 128, "mem": 1024},
]

SPOT_DISCOUNT: dict[str, float] = {
    "aws": 0.65,
    "azure": 0.60,
    "gcp": 0.70,
}

RESERVED_DISCOUNT_1YR: dict[str, float] = {
    "aws": 0.40,
    "azure": 0.35,
    "gcp": 0.30,
}

RESERVED_DISCOUNT_3YR: dict[str, float] = {
    "aws": 0.55,
    "azure": 0.50,
    "gcp": 0.45,
}


class CloudPricingService:
    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/cloud")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_pricing(self, provider: str, region: str = "us-east-1") -> list[CloudPricingRow]:
        cache_key = f"{provider}:{region}"
        if cache_key in GPU_PRICING_CACHE:
            return [CloudPricingRow(**r) for r in GPU_PRICING_CACHE[cache_key]]

        rows: list[CloudPricingRow] = []
        instances = self._get_instances(provider)
        spot_discount = SPOT_DISCOUNT.get(provider, 0.6)
        for inst in instances:
            hourly = inst["hourly"]
            row = CloudPricingRow(
                provider=provider,
                region=region,
                instance_type=inst["instance"],
                gpu_model=inst["gpu"],
                gpu_count=inst["count"],
                hourly_cost=round(hourly, 4),
                monthly_cost=round(hourly * 730, 2),
                spot_hourly=round(hourly * (1 - spot_discount), 4),
                vcpu_count=inst.get("vcpu", 0),
                memory_gb=float(inst.get("mem", 0)),
                source="static",
            )
            rows.append(row)

        GPU_PRICING_CACHE[cache_key] = [vars(r) for r in rows]
        return rows

    def _get_instances(self, provider: str) -> list[dict[str, Any]]:
        if provider == "aws":
            return AWS_GPU_INSTANCES
        elif provider == "azure":
            return AZURE_GPU_INSTANCES
        elif provider == "gcp":
            return GCP_GPU_INSTANCES
        return []

    def get_all_providers(self) -> list[str]:
        return ["aws", "azure", "gcp"]

    def compare_gpu(self, gpu_model: str, region: str = "us-east-1") -> list[CloudPricingRow]:
        results: list[CloudPricingRow] = []
        for provider in self.get_all_providers():
            pricing = self.get_pricing(provider, region)
            for p in pricing:
                if p.gpu_model.upper() == gpu_model.upper():
                    results.append(p)
        return sorted(results, key=lambda r: r.hourly_cost)

    def find_cheapest(self, gpu_model: str, gpu_count: int, region: str = "us-east-1") -> CloudPricingRow | None:
        candidates = self.compare_gpu(gpu_model, region)
        suitable = [c for c in candidates if c.gpu_count >= gpu_count and c.gpu_count % gpu_count == 0]
        if not suitable:
            suitable = candidates
        return min(suitable, key=lambda r: r.hourly_cost / r.gpu_count) if suitable else None

    def estimate_monthly_cost(self, gpu_model: str, gpu_count: int, provider: str = "aws", region: str = "us-east-1") -> float:
        pricing = self.get_pricing(provider, region)
        matching = [p for p in pricing if p.gpu_model.upper() == gpu_model.upper()]
        if not matching:
            gpu_hourly = {"H100": 5.50, "A100": 3.50, "V100": 2.50, "T4": 0.60, "L4": 0.80, "A10G": 1.20}
            rate = gpu_hourly.get(gpu_model.upper(), 2.0)
            return round(rate * gpu_count * 730, 2)
        cheapest = min(matching, key=lambda p: p.hourly_cost / p.gpu_count)
        return round(cheapest.hourly_cost / cheapest.gpu_count * gpu_count * 730, 2)

    def get_spot_savings(self, gpu_model: str, gpu_count: int, provider: str = "aws", region: str = "us-east-1") -> dict[str, Any]:
        ondemand = self.estimate_monthly_cost(gpu_model, gpu_count, provider, region)
        spot_discount = SPOT_DISCOUNT.get(provider, 0.6)
        spot = round(ondemand * (1 - spot_discount), 2)
        return {
            "provider": provider,
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
            "ondemand_monthly": ondemand,
            "spot_monthly": spot,
            "monthly_savings": round(ondemand - spot, 2),
            "savings_percent": round(spot_discount * 100, 1),
        }

    def get_reserved_savings(self, gpu_model: str, gpu_count: int, provider: str = "aws", region: str = "us-east-1") -> dict[str, Any]:
        ondemand = self.estimate_monthly_cost(gpu_model, gpu_count, provider, region)
        d1 = RESERVED_DISCOUNT_1YR.get(provider, 0.35)
        d3 = RESERVED_DISCOUNT_3YR.get(provider, 0.50)
        return {
            "provider": provider,
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
            "ondemand_monthly": ondemand,
            "reserved_1yr_monthly": round(ondemand * (1 - d1), 2),
            "reserved_3yr_monthly": round(ondemand * (1 - d3), 2),
            "reserved_1yr_savings_percent": round(d1 * 100, 1),
            "reserved_3yr_savings_percent": round(d3 * 100, 1),
            "recommendation": "3yr" if d3 > d1 else "1yr",
        }

    def get_all_provider_costs(self, gpu_model: str, gpu_count: int, region: str = "us-east-1") -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for provider in self.get_all_providers():
            od = self.estimate_monthly_cost(gpu_model, gpu_count, provider, region)
            results.append({
                "provider": provider,
                "ondemand_monthly": od,
                "spot_monthly": round(od * (1 - SPOT_DISCOUNT.get(provider, 0.6)), 2),
                "reserved_1yr_monthly": round(od * (1 - RESERVED_DISCOUNT_1YR.get(provider, 0.35)), 2),
                "reserved_3yr_monthly": round(od * (1 - RESERVED_DISCOUNT_3YR.get(provider, 0.5)), 2),
            })
        return sorted(results, key=lambda r: r["ondemand_monthly"])


class CloudPricingFetcher:
    AWS_PRICING_API = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.json"
    AZURE_RETAIL_API = "https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Virtual Machines' and armSkuName eq 'Standard_NC6s_v3'"
    GCP_PRICING_API = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"

    def __init__(self, fallback: CloudPricingService | None = None) -> None:
        self.fallback = fallback or CloudPricingService()
        self._live_data: dict[str, list[CloudPricingRow]] = {}

    def fetch_live_pricing(self, provider: str, region: str = "us-east-1") -> list[CloudPricingRow]:
        if provider in self._live_data:
            return self._live_data[provider]

        rows: list[CloudPricingRow] | None = None
        if provider == "aws":
            rows = self._fetch_aws(region)
        elif provider == "azure":
            rows = self._fetch_azure(region)
        elif provider == "gcp":
            rows = self._fetch_gcp(region)

        if rows:
            self._live_data[provider] = rows
            return rows
        return self.fallback.get_pricing(provider, region)

    def _fetch_aws(self, region: str) -> list[CloudPricingRow] | None:
        try:
            req = Request(self.AWS_PRICING_API, headers={"Accept-Encoding": "gzip"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            products = data.get("products", {})
            terms = data.get("terms", {}).get("OnDemand", {})
            rows: list[CloudPricingRow] = []
            gpu_skus = {k: v for k, v in products.items() if "gpu" in str(v.get("attributes", {})).lower()}
            for sku, product in gpu_skus.items():
                attrs = product.get("attributes", {})
                if attrs.get("regionCode") != region:
                    continue
                instance = attrs.get("instanceType", "")
                gpu_model = attrs.get("gpuModel", "")
                gpu_count = int(attrs.get("gpuCount", 0) or 0)
                if gpu_count == 0:
                    continue
                vcpu = int(attrs.get("vcpu", 0) or 0)
                mem_str = attrs.get("memory", "0")
                try:
                    mem = float(mem_str.split()[0]) if mem_str.split() else 0
                except (ValueError, IndexError):
                    mem = 0
                term_keys = list(terms.get(sku, {}).keys())
                term_key = term_keys[0] if term_keys else ""
                price_dim = terms.get(sku, {}).get(term_key, {})
                price_per_hour = 0.0
                for dim_key, dim in price_dim.get("priceDimensions", {}).items():
                    if "Hrs" in dim.get("unit", ""):
                        price_per_hour = float(dim.get("pricePerUnit", {}).get("USD", 0))
                        break
                if price_per_hour > 0:
                    rows.append(CloudPricingRow(
                        provider="aws", region=region, instance_type=instance,
                        gpu_model=gpu_model, gpu_count=gpu_count,
                        hourly_cost=round(price_per_hour, 4),
                        monthly_cost=round(price_per_hour * 730, 2),
                        vcpu_count=vcpu, memory_gb=mem, source="live_aws",
                    ))
            return rows if rows else None
        except Exception as exc:
            logger.debug("AWS live pricing fetch failed: %s", exc)
            return None

    def _fetch_azure(self, region: str) -> list[CloudPricingRow] | None:
        try:
            req = Request(self.AZURE_RETAIL_API)
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            rows: list[CloudPricingRow] = []
            for item in data.get("Items", []):
                if item.get("armRegionName") != region:
                    continue
                sku = item.get("armSkuName", "")
                gpu_str = item.get("gpu", "")
                if not gpu_str and "NC" not in sku and "ND" not in sku and "NV" not in sku:
                    continue
                instance = sku
                gpu_model = "V100" if "V100" in str(item) else ("A100" if "A100" in str(item) else gpu_str or "unknown")
                gpu_count = 0
                if gpu_str:
                    try:
                        gpu_count = int(gpu_str)
                    except ValueError:
                        gpu_count = 1
                elif "NC" in sku:
                    gpu_count = 1
                hourly = float(item.get("retailPrice", 0) or 0)
                if hourly > 0 and gpu_count > 0:
                    rows.append(CloudPricingRow(
                        provider="azure", region=region, instance_type=instance,
                        gpu_model=gpu_model, gpu_count=gpu_count,
                        hourly_cost=round(hourly, 4),
                        monthly_cost=round(hourly * 730, 2),
                        source="live_azure",
                    ))
            return rows if rows else None
        except Exception as exc:
            logger.debug("Azure live pricing fetch failed: %s", exc)
            return None

    def _fetch_gcp(self, region: str) -> list[CloudPricingRow] | None:
        try:
            req = Request(self.GCP_PRICING_API)
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            rows: list[CloudPricingRow] = []
            for sku in data.get("skus", []):
                if region not in str(sku.get("regionIds", [])):
                    continue
                desc = sku.get("description", "").lower()
                if "gpu" not in desc and "a100" not in desc and "h100" not in desc:
                    continue
                instance = sku.get("name", "unknown")
                for price_info in sku.get("pricingInfo", []):
                    for tier in price_info.get("pricingExpression", {}).get("tieredRates", []):
                        unit_price = tier.get("unitPrice", {})
                        hourly = float(unit_price.get("units", 0) or 0) / 1e9
                        if hourly > 0:
                            rows.append(CloudPricingRow(
                                provider="gcp", region=region, instance_type=instance,
                                gpu_model=(sku.get("description", "").split() or ["unknown"])[0],
                                gpu_count=1,
                                hourly_cost=round(hourly, 4),
                                monthly_cost=round(hourly * 730, 2),
                                source="live_gcp",
                            ))
                            break
            return rows if rows else None
        except Exception as exc:
            logger.debug("GCP live pricing fetch failed: %s", exc)
            return None

    def get_pricing(self, provider: str, region: str = "us-east-1") -> list[CloudPricingRow]:
        live = self.fetch_live_pricing(provider, region)
        if live:
            return live
        return self.fallback.get_pricing(provider, region)

    def get_all_providers(self) -> list[str]:
        return self.fallback.get_all_providers()

    def compare_gpu(self, gpu_model: str, region: str = "us-east-1") -> list[CloudPricingRow]:
        return self.fallback.compare_gpu(gpu_model, region)

    def find_cheapest(self, gpu_model: str, gpu_count: int, region: str = "us-east-1") -> CloudPricingRow | None:
        return self.fallback.find_cheapest(gpu_model, gpu_count, region)

    def estimate_monthly_cost(self, gpu_model: str, gpu_count: int, provider: str = "aws", region: str = "us-east-1") -> float:
        return self.fallback.estimate_monthly_cost(gpu_model, gpu_count, provider, region)

    def get_spot_savings(self, gpu_model: str, gpu_count: int, provider: str = "aws", region: str = "us-east-1") -> dict[str, Any]:
        return self.fallback.get_spot_savings(gpu_model, gpu_count, provider, region)

    def get_reserved_savings(self, gpu_model: str, gpu_count: int, provider: str = "aws", region: str = "us-east-1") -> dict[str, Any]:
        return self.fallback.get_reserved_savings(gpu_model, gpu_count, provider, region)

    def get_all_provider_costs(self, gpu_model: str, gpu_count: int, region: str = "us-east-1") -> list[dict[str, Any]]:
        return self.fallback.get_all_provider_costs(gpu_model, gpu_count, region)
