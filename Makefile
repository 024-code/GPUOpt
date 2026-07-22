SHELL := /bin/bash
PYTHON ?= python
SAST_REPORT ?= bandit-results.sarif
TRIVY_REPORT ?= trivy-results.sarif

.PHONY: install dev test lint seed check-all migrate migrate-sql migrate-stamp
.PHONY: migrate-new migrate-history migrate-current
.PHONY: compose-up compose-down kind-up kind-down port-forward
.PHONY: sast deps-scan secrets-scan container-scan sbom security-all

install:
	$(PYTHON) -m pip install -e '.[dev]'

dev:
	PYTHONPATH=src uvicorn gpuopt.main:app --reload --host 0.0.0.0 --port 8080

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests

seed:
	PYTHONPATH=src $(PYTHON) -m gpuopt.cli seed --file environments.mock.yaml

check-all:
	PYTHONPATH=src $(PYTHON) -m gpuopt.cli check-all --file environments.mock.yaml

migrate:
	PYTHONPATH=src $(PYTHON) scripts/migrate.py --alembic

migrate-sql:
	PYTHONPATH=src $(PYTHON) scripts/migrate.py --sql

migrate-stamp:
	PYTHONPATH=src $(PYTHON) scripts/migrate.py --stamp

migrate-new:
	@read -p "Migration message: " msg; \
	PYTHONPATH=src $(PYTHON) -m alembic -c alembic.ini revision --autogenerate -m "$$msg"

migrate-history:
	PYTHONPATH=src $(PYTHON) -m alembic -c alembic.ini history

migrate-current:
	PYTHONPATH=src $(PYTHON) -m alembic -c alembic.ini current

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down -v

kind-up:
	./scripts/bootstrap_kind.sh

kind-down:
	kind delete cluster --name gpuopt

port-forward:
	kubectl --context kind-gpuopt -n gpuopt-system port-forward service/gpuopt-backend 8080:8080

# ── Security targets ──────────────────────────────────────────

sast:
	$(PYTHON) -m bandit -c .bandit -r src/ -f sarif -o $(SAST_REPORT)

deps-scan:
	pip-audit --strict --progress-spinner off

secrets-scan:
	gitleaks detect --source . --no-git --verbose

container-scan:
	docker build -t gpuopt-backend:scan .
	trivy image --severity HIGH,CRITICAL --ignore-unfixed gpuopt-backend:scan

sbom:
	docker build -t gpuopt-backend:sbom .
	trivy image --format cyclonedx --output gpuopt-backend.sbom.json gpuopt-backend:sbom

security-all: sast deps-scan
