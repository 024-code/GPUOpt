from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import APIRouter, Query

from ..config import get_settings
from ..registry import get_registry
from .evolution import PolicyEvolutionEngine
from .gatekeeper import GatekeeperDeployer

logger = logging.getLogger(__name__)

policy_router = APIRouter(prefix="/api/v1/policy", tags=["policy-evolution"])


def _get_engine() -> PolicyEvolutionEngine:
    reg = get_registry()
    return reg.get_or_create("policy_engine", PolicyEvolutionEngine)


def _get_deployer() -> GatekeeperDeployer:
    settings = get_settings()
    reg = get_registry()
    existing = reg.get("gatekeeper_deployer")
    if existing is not None:
        return existing
    deployer = GatekeeperDeployer(
        base_url=settings.gatekeeper_api_url,
        dry_run=not settings.gatekeeper_enabled,
    )
    reg.register("gatekeeper_deployer", deployer, force=True)
    return deployer


@policy_router.post("/evolve")
def evolve_policies(metrics: List[dict]) -> dict:
    engine = _get_engine()
    best = engine.evolve(metrics)
    return {
        "status": "evolution_complete",
        "generations": engine.generations,
        "best_fitness": best.fitness_score,
        "policy_rego": best.to_rego(),
    }


@policy_router.get("/best-policy")
def get_best_policy() -> dict:
    engine = _get_engine()
    if engine.best_chromosome is not None:
        return {
            "status": "success",
            "policy": engine.get_best_policy(),
            "fitness": engine.best_chromosome.fitness_score,
            "generation": engine.best_chromosome.generation,
        }
    return {"status": "no_policy_evolved_yet"}


@policy_router.post("/deploy-policy")
def deploy_policy(gatekeeper_url: str = Query("", description="Overrides the configured Gatekeeper API URL")) -> dict:
    engine = _get_engine()
    if engine.best_chromosome is None:
        return {"status": "error", "reason": "No evolved policy available"}

    rego = engine.get_best_policy()
    policy_yaml = """
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: evolvedgpupolicy
spec:
  crd:
    spec:
      names:
        kind: EvolvedGPUPolicy
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
""" + "\n".join(
        f"        {line}" for line in rego.split("\n")
    )

    if gatekeeper_url:
        deployer = GatekeeperDeployer(base_url=gatekeeper_url, dry_run=False)
    else:
        deployer = _get_deployer()

    deploy_result = deployer.deploy(policy_yaml)

    return {
        "status": deploy_result.get("status", "policy_deployed"),
        "template": policy_yaml,
        "deploy_result": deploy_result,
    }


@policy_router.get("/gatekeeper-health")
def gatekeeper_health() -> dict:
    deployer = _get_deployer()
    return deployer.health_check()
