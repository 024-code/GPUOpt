from __future__ import annotations

import json
import logging
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class GatekeeperDeployer:
    def __init__(self, base_url: str = "", dry_run: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run

    @property
    def enabled(self) -> bool:
        return bool(self.base_url) and not self.dry_run

    def deploy(self, constraint_template_yaml: str, constraint_name: str = "evolvedgpupolicy") -> dict[str, Any]:
        if self.dry_run:
            logger.info("Dry-run: would deploy constraint template to %s", self.base_url or "(no endpoint)")
            return {
                "status": "dry_run",
                "message": "Deployment skipped (dry-run mode)",
                "template": constraint_template_yaml,
            }

        if not self.base_url:
            return {
                "status": "error",
                "reason": "No Gatekeeper API URL configured",
                "template": constraint_template_yaml,
            }

        result = self._apply_template(constraint_template_yaml)
        result["template"] = constraint_template_yaml
        return result

    def _apply_template(self, yaml_content: str) -> dict[str, Any]:
        url = f"{self.base_url}/v1/constrainttemplates"
        payload = {
            "apiVersion": "templates.gatekeeper.sh/v1",
            "kind": "ConstraintTemplate",
            "metadata": {"name": "evolvedgpupolicy"},
            "spec": {"crd": {"spec": {"names": {"kind": "EvolvedGPUPolicy"}}}, "targets": [{"target": "admission.k8s.gatekeeper.sh", "rego": yaml_content}]},
        }
        try:
            req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="PUT")
            with urlopen(req, timeout=10) as resp:
                body = resp.read().decode()
                logger.info("Gatekeeper constraint template applied: %s", resp.status)
                return {"status": "deployed", "api_status": resp.status, "response": body[:500]}
        except Exception as exc:
            logger.error("Gatekeeper deploy failed: %s", exc)
            return {"status": "deploy_failed", "error": str(exc)}

    def health_check(self) -> dict[str, Any]:
        if not self.base_url:
            return {"reachable": False, "reason": "No Gatekeeper URL configured"}
        try:
            req = Request(f"{self.base_url}/v1/healthz", method="GET")
            with urlopen(req, timeout=5) as resp:
                return {"reachable": resp.status == 200, "status_code": resp.status}
        except Exception as exc:
            return {"reachable": False, "error": str(exc)}
