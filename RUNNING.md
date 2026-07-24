# Running GPUOpt — Frontend & Backend

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        Docker Host                            │
│                                                              │
│  ┌───────────────────┐        ┌────────────────────────┐     │
│  │   nginx (lb)       │        │  PostgreSQL (db)       │     │
│  │   port 8080 → 80   │        │  port 5432             │     │
│  └────────┬──────────┘        └────────────────────────┘     │
│           │                                                  │
│  ┌────────▼──────────┐        ┌────────────────────────┐     │
│  │  GPUOpt API x3    │◄──────►│  vLLM Server           │     │
│  │  (api-a/b/c)      │        │  port 8001             │     │
│  │  port 8080        │        │  GPU-accelerated       │     │
│  └────────┬──────────┘        └────────────────────────┘     │
│           │                                                  │
│  ┌────────▼──────────┐        ┌────────────────────────┐     │
│  │  Prometheus        │        │  Grafana               │     │
│  │  port 9090         │        │  port 3000             │     │
│  └───────────────────┘        └────────────────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │  React Frontend (dev mode, port 5173)            │        │
│  │  or served statically by GPUOpt API (port 8080)  │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Requirement | Version |
|---|---|
| Docker Desktop | 24+ with WSL2 backend |
| NVIDIA Driver | 535+ (on host) |
| NVIDIA Container Toolkit | installed |
| NVIDIA GPU | 8GB+ VRAM (RTX 3060+, A-series) |
| Node.js | 18+ (for frontend dev mode) |
| Python | 3.12+ (for local backend dev) |

---

## Running the Backend

### Option A: Full Stack with Docker Compose (Production-like)

```bash
cd GPUOpt

# Build and start all services
docker compose up --build -d
```

This starts:
| Service | Internal Port | External Port | Purpose |
|---|---|---|---|
| `lb` (nginx) | 80 | 8080 | Load balancer across API instances |
| `api-a/b/c` | 8080 | — | GPUOpt API (3 replicas) |
| `db` | 5432 | 5432 | PostgreSQL |
| `prometheus` | 9090 | 9090 | Metrics collection |
| `grafana` | 3000 | 3000 | Monitoring dashboards |

Verify:
```bash
docker compose ps
curl http://localhost:8080/health/ready
```

### Option B: Local Development (with vLLM)

```bash
cd GPUOpt

# Start GPUOpt API + vLLM with GPU support
docker compose -f docker-compose.local.yml up --build -d
```

This starts:
| Service | Port | Purpose |
|---|---|---|
| `api` | 8080 | GPUOpt backend (GPU monitor + model mgmt) |
| `vllm` | 8001 | vLLM OpenAI-compatible inference server |

Verify:
```bash
curl http://localhost:8080/health/ready
curl http://localhost:8001/v1/models
```

### Option C: Backend Only (No Docker, Dev Mode)

```bash
cd GPUOpt

# Install dependencies
python -m pip install -e '.[dev]'

# Copy and configure environment
cp .env.example .env

# Run database migrations
make migrate

# Seed mock data (optional)
make seed

# Start dev server (hot reload)
make dev
```

The API will be available at `http://localhost:8080`.

---

## Running the Frontend

The frontend is a React + Vite + TypeScript app in `frontend-react/`.

### Option A: Static Build (Served by Backend)

```bash
cd GPUOpt/frontend-react

npm install
npm run build
```

This produces a static bundle in `frontend-react/dist/`. If the backend's `GPUOPT_FRONTEND_DIR` env points to this directory (default: `/app/frontend`), the API serves the frontend at `http://localhost:8080`.

After building, restart the API container if running in Docker:
```bash
docker compose -f docker-compose.local.yml restart api
```

### Option B: Development Mode (Hot Reload)

```bash
cd GPUOpt/frontend-react

npm install
npm run dev
```

Opens at `http://localhost:5173`.

The Vite dev server proxies `/api` requests to `http://localhost:8080` and `/ws` WebSocket connections to `ws://localhost:8080` (configured in `vite.config.ts`).

---

## Environment Variables

### Backend (.env)

| Variable | Default | Description |
|---|---|---|
| `GPUOPT_ENV` | `development` | Runtime environment |
| `GPUOPT_DATABASE_PATH` | `./data/gpuopt.db` | SQLite database path |
| `GPUOPT_DATABASE_URL` | — | PostgreSQL connection string (overrides SQLite) |
| `GPUOPT_LOG_LEVEL` | `INFO` | Logging verbosity |
| `GPUOPT_ALLOW_MOCK_GPU` | `true` | Enable mock GPU data when no GPU detected |
| `GPUOPT_API_HOST` | `0.0.0.0` | API bind address |
| `GPUOPT_API_PORT` | `8080` | API port |
| `GPUOPT_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `GPUOPT_API_KEY` | — | API key for authentication |
| `GPUOPT_DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `GPUOPT_RTX_PARTITIONS_GB` | — | GPU VRAM partitioning (e.g. `"6,6,6,6"`) |
| `HUGGING_FACE_HUB_TOKEN` | — | HF token for gated models (used by vLLM) |

### Frontend Proxy (vite.config.ts)

The Vite dev server proxies:
- `/api` → `http://localhost:8080` (REST API calls)
- `/ws` → `ws://localhost:8080` (WebSocket for live GPU telemetry)

---

## Useful Commands

### Backend

```bash
# Start full stack
docker compose up --build -d

# Start local dev with vLLM
docker compose -f docker-compose.local.yml up --build -d

# Stop services
docker compose down -v

# View logs
docker compose logs -f api-a

# Rebuild single service
docker compose up --build -d api-a

# Run tests
make test

# Run linter
make lint

# Run backend locally (no Docker)
make dev
```

### Frontend

```bash
# Install dependencies
cd frontend-react && npm install

# Dev server with hot reload
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

---

## Ports Reference

| Port | Service | Mode |
|---|---|---|
| 8080 | GPUOpt API (or nginx LB) | All |
| 5173 | React dev server | Frontend dev |
| 5432 | PostgreSQL | Full stack |
| 8001 | vLLM inference server | Local dev |
| 9090 | Prometheus | Full stack |
| 3000 | Grafana | Full stack |

---

## Troubleshooting

| Problem | Check |
|---|---|
| API not responding | `docker compose ps`, `curl localhost:8080/health/ready` |
| Frontend shows blank page | Check browser console for proxy errors; ensure backend is running on port 8080 |
| vLLM exits immediately | `docker compose logs vllm`; likely OOM or missing HF token |
| No GPU data in UI | `curl localhost:8080/api/v1/monitoring/gpu/snapshot`; verify GPU passthrough in Docker |
| CORS errors | Verify `GPUOPT_CORS_ORIGINS` includes frontend origin |
