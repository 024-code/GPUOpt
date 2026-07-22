# Security

## Design Principles

1. **Read-only by default**: All Kubernetes operations are non-mutating. No create, update, or delete on any K8s resource.
2. **Least privilege**: The ClusterRole only grants the minimum permissions needed for environment checks.
3. **No secrets in the database**: `kubeconfig_path` is a file path reference, not inline content. The `reject_inline_secret_data` validator rejects values containing `apiVersion:`.
4. **No mutation until production hardening**: Do not add write permissions until the recommendation engine, policy checks, approvals, action idempotency, and rollback controls have passed staging acceptance tests.

## RBAC Configuration

### ClusterRole: `gpuopt-readonly`

```yaml
rules:
  - apiGroups: [""]
    resources: ["nodes", "pods", "services", "endpoints", "events", "namespaces"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "daemonsets", "statefulsets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apiextensions.k8s.io"]
    resources: ["customresourcedefinitions"]
    verbs: ["get", "list"]
  - apiGroups: ["authorization.k8s.io"]
    resources: ["selfsubjectaccessreviews"]
    verbs: ["create"]
```

**Bound to:** `ServiceAccount/gpuopt-backend` in `gpuopt-system` namespace.

### What is NOT allowed

- No `create`, `update`, `patch`, `delete` on any resource.
- No access to `secrets`, `configmaps` (except via envFrom in the Deployment).
- No access to `persistentvolumes`, `storageclasses`, or other infrastructure resources.

## Container Security

The production Deployment applies:

| Setting | Value |
|---------|-------|
| `runAsNonRoot` | `true` |
| `runAsUser` | `10001` |
| `fsGroup` | `10001` |
| `readOnlyRootFilesystem` | `true` |
| `allowPrivilegeEscalation` | `false` |
| `capabilities.drop` | `["ALL"]` |

- The container runs as a non-root user with no Linux capabilities.
- The root filesystem is read-only; only `/data` and `/tmp` are writable (emptyDir volumes).
- The Dockerfile uses a non-root user (`gpuopt`, UID 10001).

## Data Sensitivity

| Data | Sensitivity | Storage |
|------|-------------|---------|
| Cluster metadata (name, environment, connector type) | Low | SQLite/PostgreSQL |
| kubeconfig paths | Medium | SQLite/PostgreSQL (references only, not content) |
| Check reports (status, latency, details) | Low | SQLite/PostgreSQL |
| API server version info | Low | Check report details |
| Node names and capacity | Low | Check report details |
| GPU model labels | Low | Check report details |

**Never store:**
- Kubeconfig file contents
- Bearer tokens
- TLS certificates
- ServiceAccount token data

## Network Security

- The API server listens on `0.0.0.0:8080` by default. In production, use a Service with appropriate network policies.
- The `/metrics` endpoint is unauthenticated. Protect it with network policies if metrics are sensitive.
- The `/docs` and `/redoc` endpoints expose the full API schema. Disable in production by setting `docs_url=None` and `redoc_url=None` in the FastAPI app constructor.

## Environment Isolation

Clusters are tagged by environment (`sandbox`, `development`, `staging`, `production`). The environment summary endpoint provides per-environment health visibility.

**Recommended practice:**
- Register production clusters with `options.allow_mock_gpu: false`.
- Use separate kubeconfig contexts per environment.
- Use separate ServiceAccounts per environment in multi-tenant clusters.

## Audit Trail

All check reports are persisted with:
- Timestamps (`started_at`, `completed_at`)
- Overall status
- Full check results (JSON)
- Cluster reference (foreign key)

Production should add:
- User/identity who triggered the check
- IP address logging
- Retention policies for check history
- Export to centralized audit log (SIEM)

## Production Hardening Checklist

- [x] Add authentication (RBAC with API keys or keyless mode)
- [x] Add rate limiting (per-IP in-memory sliding window)
- [x] Add request logging and audit trail (`AuditMiddleware` + `AuditStore`)
- [x] Enable read-only K8s RBAC (ClusterRole: `gpuopt-readonly`)
- [x] Container security (non-root user, readOnlyRootFilesystem, drop all caps)
- [x] SAST scanning (Bandit) integrated in CI
- [x] Dependency auditing (pip-audit) integrated in CI
- [x] Secrets scanning (Gitleaks) integrated in CI
- [x] Container scanning (Trivy) integrated in CI
- [x] IaC scanning (Checkov) integrated in CI
- [x] SBOM generation in CI (Syft + CycloneDX)
- [x] Docker metadata labels (OCI annotations)
- [x] HEALTHCHECK instruction in Dockerfile
- [x] `.dockerignore` to exclude build-context leaks
- [x] `POSTGRES_PASSWORD` via env var (not hardcoded)
- [ ] Replace SQLite with PostgreSQL
- [ ] Configure TLS termination (ingress or load balancer)
- [ ] Set up network policies in K8s
- [ ] Disable Swagger UI in production
- [ ] Set up alerting on `gpuopt_cluster_health_status == 3` (PrometheusAlertManager)
- [ ] Implement secret rotation for kubeconfig references
- [ ] Add Pod Security Standards (restricted profile)
- [ ] Add OpenID Connect / OAuth2 integration
- [ ] Add DAST scanning (OWASP ZAP)
