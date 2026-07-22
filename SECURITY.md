# Security Policy

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| latest  | :white_check_mark: |

## Reporting a Vulnerability

Open an issue at https://github.com/anomalyco/GPUOpt/issues
or contact the maintainers directly.

## Security Tooling

This project uses the following automated security scanning:

| Tool      | Scope              | CI |
|-----------|--------------------|----|
| Bandit    | Python SAST        | Yes|
| pip-audit | Dependency audit   | Yes|
| Gitleaks  | Secrets scanning   | Yes|
| Trivy     | Container + IaC    | Yes|
| Checkov   | K8s manifest scan  | Yes|

See [docs/SECURITY.md](docs/SECURITY.md) for the full hardening guide.
