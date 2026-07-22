#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-gpuopt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for cmd in docker kind kubectl; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing required command: $cmd" >&2; exit 1; }
done

echo "[1/7] Creating kind cluster: ${CLUSTER_NAME}"
if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  kind create cluster --name "${CLUSTER_NAME}" --config "${ROOT_DIR}/infra/kind/kind-config.yaml"
fi

echo "[2/7] Labelling mock GPU nodes"
workers=( $(kubectl --context "kind-${CLUSTER_NAME}" get nodes -o name | grep worker || true) )
for i in "${!workers[@]}"; do
  model="mock-a100"
  count="4"
  if [[ "$i" -eq 1 ]]; then model="mock-l40s"; count="4"; fi
  kubectl --context "kind-${CLUSTER_NAME}" label --overwrite "${workers[$i]}" \
    gpuopt.ai/mock-gpu-count="${count}" gpuopt.ai/gpu-model="${model}"
done

echo "[3/7] Building backend image"
docker build -t gpuopt-backend-sandbox:local "${ROOT_DIR}"
kind load docker-image --name "${CLUSTER_NAME}" gpuopt-backend-sandbox:local

echo "[4/7] Applying namespace, RBAC and mock telemetry"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/base/namespace.yaml"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/base/serviceaccount.yaml"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/base/rbac.yaml"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/mock-dcgm/configmap.yaml"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/mock-dcgm/deployment.yaml"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/mock-dcgm/service.yaml"

echo "[5/7] Deploying GPUOpt backend"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/base/configmap.yaml"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/base/deployment.yaml"
kubectl --context "kind-${CLUSTER_NAME}" apply -f "${ROOT_DIR}/infra/k8s/base/service.yaml"

echo "[6/7] Waiting for workloads"
kubectl --context "kind-${CLUSTER_NAME}" -n gpuopt-system rollout status deployment/mock-dcgm-exporter --timeout=120s
kubectl --context "kind-${CLUSTER_NAME}" -n gpuopt-system rollout status deployment/gpuopt-backend --timeout=120s

echo "[7/7] Registering in-cluster environment and running checks"
kubectl --context "kind-${CLUSTER_NAME}" -n gpuopt-system port-forward service/gpuopt-backend 8080:8080 >/tmp/gpuopt-port-forward.log 2>&1 &
PF_PID=$!
trap 'kill ${PF_PID} 2>/dev/null || true' EXIT
sleep 3
curl -fsS -X PUT "http://127.0.0.1:8080/api/v1/clusters/by-name/local-kind" \
  -H 'content-type: application/json' \
  -d '{"name":"local-kind","environment":"sandbox","connector_type":"kubernetes","description":"Kind cluster using in-cluster credentials and mock GPU labels","in_cluster":true,"options":{"allow_mock_gpu":true}}' >/tmp/gpuopt-register.json
CLUSTER_ID=$(python -c 'import json; print(json.load(open("/tmp/gpuopt-register.json"))["id"])')
curl -fsS -X POST "http://127.0.0.1:8080/api/v1/clusters/${CLUSTER_ID}/checks" | python -m json.tool

echo
printf 'Sandbox is ready. Run:\n  kubectl --context kind-%s -n gpuopt-system port-forward service/gpuopt-backend 8080:8080\nThen open http://127.0.0.1:8080/docs\n' "${CLUSTER_NAME}"
