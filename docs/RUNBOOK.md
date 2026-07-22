# Operational Runbook

## Overall Status Definitions

| Status | Meaning |
|--------|---------|
| `PASS` | All mandatory checks pass and optional integrations are available |
| `WARN` | Cluster is usable for read-only development, but one or more optional components or mock resources are in use |
| `FAIL` | API access, RBAC, node readiness, real GPU discovery, or mandatory telemetry is unavailable |
| `SKIP` | All checks were skipped (connector not configured) |

---

## Common Issues and Remediations

### Kubernetes Configuration Failure

**Symptom:** Check returns `kubernetes_config` FAIL with "Unable to load Kubernetes credentials".

**Cause:** Invalid kubeconfig path, wrong context name, or missing ServiceAccount token.

**Remediation:**
- Verify the kubeconfig file exists and the context is valid: `kubectl config get-contexts`.
- If using `in_cluster=true`, confirm the pod has a mounted ServiceAccount token.
- Do **not** store kubeconfig content or bearer tokens in the GPUOpt database.

---

### RBAC Failure

**Symptom:** Check returns `rbac_permissions` FAIL with "Required RBAC permissions are missing".

**Cause:** The ServiceAccount lacks the necessary read-only permissions.

**Remediation:**
- Apply the least-privilege ClusterRole: `kubectl apply -f infra/k8s/base/rbac.yaml`.
- Required permissions: list nodes, list pods, list customresourcedefinitions.
- Keep mutation rights in a separate role reserved for the future actuator.

---

### No GPU Resources

**Symptom:** Check returns `gpu_inventory` FAIL with "No NVIDIA GPU extended resources were detected".

**On a real cluster:**
1. Verify the NVIDIA driver is installed: `nvidia-smi` on the node.
2. Verify the device plugin or GPU Operator is running.
3. Confirm nodes advertise `nvidia.com/gpu`: `kubectl get node -o json | jq '.items[].status.capacity'`.

**On kind:**
1. Label worker nodes with mock GPU labels:
   ```bash
   kubectl label node <worker> gpuopt.ai/mock-gpu-count=4 gpuopt.ai/gpu-model=mock-a100
   ```
2. Set `options.allow_mock_gpu: true` when registering the cluster.

---

### DCGM Exporter Missing

**Symptom:** Check returns `dcgm_exporter` FAIL with "DCGM exporter pods were not detected".

**Remediation:**
1. Install the NVIDIA GPU Operator (includes DCGM exporter by default).
2. Or deploy the DCGM exporter manually as a DaemonSet on GPU nodes.
3. Verify the exporter service exists: `kubectl get svc -A | grep dcgm`.
4. Confirm the `/metrics` endpoint is reachable.

---

### Prometheus Missing

**Symptom:** Check returns `prometheus` WARN with "No Prometheus service was discovered".

**Remediation:**
1. Install Prometheus Operator or deploy Prometheus manually.
2. Ensure a service with "prometheus" in its name exists in `monitoring`, `prometheus`, or `observability` namespace.
3. For external Prometheus, configure the endpoint in the monitoring pipeline.

---

### GPU Operator Missing (Optional)

**Symptom:** Check returns `gpu_operator` WARN with "NVIDIA GPU Operator pods were not detected".

This is optional. The GPU Operator is recommended but not required if the device plugin is installed directly.

---

### Batch Scheduler Missing (Optional)

**Symptom:** Check returns `batch_scheduler` WARN with "No Kueue or Volcano CRDs detected".

This is optional. Install Kueue or Volcano if queued/gang-scheduled AI workloads are required.

---

## Monitoring

### Prometheus Metrics

| Metric | What it tells you |
|--------|-------------------|
| `gpuopt_environment_check_runs_total` | How many checks have run, grouped by cluster/environment/status |
| `gpuopt_environment_check_duration_seconds` | How long checks take (histogram) |
| `gpuopt_cluster_health_status` | Current health of each cluster (0=unchecked, 1=healthy, 2=warning, 3=failing) |

### Useful Prometheus Queries

```promql
# Check success rate
rate(gpuopt_environment_check_runs_total{status="pass"}[5m])
  / rate(gpuopt_environment_check_runs_total[5m])

# Failing clusters
gpuopt_cluster_health_status == 3

# Average check duration
rate(gpuopt_environment_check_duration_seconds_sum[5m])
  / rate(gpuopt_environment_check_duration_seconds_count[5m])
```

---

## Database Management

The SQLite database is stored at the path configured by `GPUOPT_DATABASE_PATH`.

### Backup

```bash
cp data/gpuopt.db data/gpuopt.db.backup
```

### Reset

```bash
rm data/gpuopt.db
# The database is auto-created on next startup
```

### Inspect

```bash
sqlite3 data/gpuopt.db ".tables"
sqlite3 data/gpuopt.db "SELECT id, name, environment FROM clusters;"
sqlite3 data/gpuopt.db "SELECT id, cluster_id, overall_status, completed_at FROM check_reports ORDER BY completed_at DESC LIMIT 10;"
```

---

## Troubleshooting Checklist

| Step | Command |
|------|---------|
| Verify API is running | `curl http://127.0.0.1:8080/health/ready` |
| Check logs | `docker compose logs api` or `kubectl -n gpuopt-system logs -l app.kubernetes.io/name=gpuopt-backend` |
| Verify cluster is registered | `curl http://127.0.0.1:8080/api/v1/clusters` |
| Run a single check | `curl -X POST http://127.0.0.1:8080/api/v1/clusters/{id}/checks` |
| View environment summary | `curl http://127.0.0.1:8080/api/v1/environments/summary` |
| Check Prometheus metrics | `curl http://127.0.0.1:8080/metrics` |
| Verify K8s connectivity | `kubectl cluster-info` |
| Verify RBAC | `kubectl auth can-i list nodes --as=system:serviceaccount:gpuopt-system:gpuopt-backend` |
