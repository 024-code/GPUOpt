#!/usr/bin/env bash
set -euo pipefail

GPUOPT_URL="${GPUOPT_URL:-http://127.0.0.1:8080}"
CONFIG_FILE="${1:-examples/benchmark-real.json}"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Usage: GPUOPT_URL=http://127.0.0.1:8080 $0 <config-file>"
  echo "Config file not found: $CONFIG_FILE"
  exit 1
fi

echo "============================================"
echo " GPUOpt Inference Benchmark"
echo "============================================"
echo "Target:    $GPUOPT_URL"
echo "Config:    $CONFIG_FILE"
echo ""

START=$(date +%s%N)

RESPONSE=$(curl -s -X POST "$GPUOPT_URL/api/v1/inference/benchmark" \
  -H "Content-Type: application/json" \
  --data-binary @"$CONFIG_FILE")

END=$(date +%s%N)
RUNTIME_MS=$(( (END - START) / 1000000 ))

echo "$RESPONSE" | python3 -m json.tool

echo ""
echo "Client-side wall time: ${RUNTIME_MS}ms"
echo ""

if echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('errors', 0) > 0:
    sys.exit(1)
if d.get('num_requests', 0) == 0:
    sys.exit(1)
" 2>/dev/null; then
  echo "Benchmark completed successfully."
else
  echo "Benchmark reported errors or no requests."
  exit 1
fi
