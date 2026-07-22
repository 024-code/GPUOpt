from __future__ import annotations

from typing import Any

import yaml

from .models import InferenceServiceSpec, ManifestResponse


def generate_manifests(spec: InferenceServiceSpec) -> ManifestResponse:
    tp = spec.tensor_parallelism
    pp = spec.pipeline_parallelism
    gpu_count = tp * pp
    image = f"{spec.image}" if not spec.tag else f"{spec.image}:{spec.tag}"

    env_vars: list[dict[str, str]] = [
        {"name": "MODEL_NAME", "value": spec.model_name},
        {"name": "TENSOR_PARALLEL_SIZE", "value": str(tp)},
        {"name": "PIPELINE_PARALLEL_SIZE", "value": str(pp)},
        {"name": "MAX_NUM_SEQS", "value": "256"},
    ] + spec.env

    resources: dict[str, Any] = {
        "limits": {
            **spec.resources.limits,
            "cpu": spec.resources.limits.get("cpu", "16"),
            "memory": spec.resources.limits.get("memory", "100Gi"),
        },
    }
    if spec.resources.requests:
        resources["requests"] = spec.resources.requests

    if gpu_count > 0:
        resources["limits"]["nvidia.com/gpu"] = str(gpu_count)

    container: dict[str, Any] = {
        "name": "inference-engine",
        "image": image,
        "ports": spec.ports,
        "env": env_vars,
        "resources": resources,
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": spec.ports[0]["containerPort"]},
            "initialDelaySeconds": 60,
            "periodSeconds": 10,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": spec.ports[0]["containerPort"]},
            "initialDelaySeconds": 120,
            "periodSeconds": 30,
        },
    }
    if spec.command:
        container["command"] = spec.command
    if spec.args:
        container["args"] = spec.args

    pod_spec: dict[str, Any] = {
        "containers": [container],
        "nodeSelector": {
            "gpuopt.ai/gpu-model": "mock-a100",
            **spec.node_selector,
        },
        "tolerations": spec.tolerations or [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
        ],
    }

    deployment: dict[str, Any] = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{spec.model_name}-inference",
            "namespace": spec.namespace,
            "labels": {
                "app": spec.model_name,
                "component": "inference",
                "gpuopt.ai/managed": "true",
            },
        },
        "spec": {
            "replicas": spec.replicas,
            "selector": {"matchLabels": {"app": spec.model_name, "component": "inference"}},
            "template": {
                "metadata": {"labels": {"app": spec.model_name, "component": "inference"}},
                "spec": pod_spec,
            },
        },
    }

    service: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{spec.model_name}-inference",
            "namespace": spec.namespace,
            "labels": {"app": spec.model_name, "component": "inference"},
        },
        "spec": {
            "type": spec.service_type,
            "ports": [
                {
                    "port": spec.ports[0]["containerPort"],
                    "targetPort": spec.ports[0]["containerPort"],
                    "name": spec.ports[0].get("name", "http"),
                }
            ],
            "selector": {"app": spec.model_name, "component": "inference"},
        },
    }

    manifests: dict[str, str] = {
        "deployment.yaml": yaml.dump(deployment, default_flow_style=False, sort_keys=False),
        "service.yaml": yaml.dump(service, default_flow_style=False, sort_keys=False),
    }

    if spec.enable_hpa:
        hpa: dict[str, Any] = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{spec.model_name}-inference",
                "namespace": spec.namespace,
                "labels": {"app": spec.model_name, "component": "inference"},
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{spec.model_name}-inference",
                },
                "minReplicas": spec.replicas,
                "maxReplicas": spec.hpa_max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": spec.hpa_target_cpu_utilization}},
                    }
                ],
            },
        }
        manifests["hpa.yaml"] = yaml.dump(hpa, default_flow_style=False, sort_keys=False)

    return ManifestResponse(manifests=manifests)
