#!/usr/bin/env bash
set -euo pipefail

API_URL="${GPUOPT_API_URL:-http://127.0.0.1:8080}"

echo "GPUOpt environment preflight"
for cmd in python curl; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "PASS command: $cmd"
  else
    echo "FAIL command: $cmd" >&2
    exit 1
  fi
done

for optional in docker kubectl kind helm; do
  if command -v "$optional" >/dev/null 2>&1; then
    echo "PASS optional command: $optional"
  else
    echo "WARN optional command missing: $optional"
  fi
done

curl -fsS "${API_URL}/health/ready" | python -m json.tool
curl -fsS -X POST "${API_URL}/api/v1/environments/check-all" | python -m json.tool
