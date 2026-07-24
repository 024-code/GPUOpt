# SLM Inference with vLLM + GPUOpt — Setup Guide

This guide walks you through running **3B/4B Small Language Models (SLMs)** via **vLLM** inside Docker, with live GPU telemetry streamed to the GPUOpt dashboard.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Docker Compose (local)                                  │
│                                                          │
│  ┌──────────────┐     ┌──────────────┐                   │
│  │  GPUOpt API   │────▶│  vLLM Server │                   │
│  │  (port 8080)  │     │  (port 8001) │                   │
│  │               │     │              │                   │
│  │  GPU Monitor  │     │  SLM Model   │                   │
│  │  (pynvml)     │     │  (3B/4B)     │                   │
│  └──────┬───────┘     └──────────────┘                   │
│         │                                                │
│         │ WebSocket (live GPU stats)                     │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │  React Frontend│  (port 5173)                         │
│  │  - Model Select│                                      │
│  │  - Chat UI     │                                      │
│  │  - GPU Gauges  │                                      │
│  └────────────────┘                                      │
└──────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version / Notes |
|---|---|
| **Docker Desktop** | 24+ with WSL2 backend |
| **NVIDIA Driver** | 535+ (on **host**, not inside WSL) |
| **NVIDIA Container Toolkit** | `nvidia-ctk` installed |
| **GPU** | NVIDIA GPU with ≥8 GB VRAM (RTX 3060+, A-series) |
| **Git** | Any recent version |
| **Node.js** (optional dev) | 18+ for frontend dev server |

### Windows-Specific Checks

```powershell
# 1. Verify NVIDIA driver on host
nvidia-smi

# 2. Verify Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 3. Enable GPU support in Docker Desktop
#    Settings → Resources → WSL Integration → Enable with your WSL distro
#    Settings → Docker Engine → add "nvidia" to "runtimes"
```

---

## Step 1: Clone & Prepare

```bash
git clone <your-repo> gpuopt
cd gpuopt
```

---

## Step 2: Model Selection — Where Models Live

vLLM downloads models from **Hugging Face Hub** automatically on first run.

### Recommended 3B–4B Models

| Model | Parameters | VRAM Needed | Quality |
|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~6 GB | ★★★★★ |
| `Qwen/Qwen2.5-7B-Instruct` (fallback) | 7B | ~14 GB | ★★★★★ |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ~8 GB | ★★★★ |
| `google/gemma-2-2b-it` | 2B | ~4 GB | ★★★★ |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~6 GB | ★★★★★ |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ~4 GB | ★★★ |

> **Note:** For `meta-llama/*` models you need a Hugging Face token (see Step 3).

### Offline / Pre-Downloaded Models

If you want to avoid re-downloading, pull once and mount a cache volume:

```bash
# Pre-download on host
docker run --rm -v "d:\model-cache:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')"
```

Then mount `d:\model-cache:/root/.cache/huggingface` in the container.

---

## Step 3: Configure Hugging Face Token (for Gated Models)

If using Llama or other gated models, create `.env` in the project root:

```
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

> Get your token at https://huggingface.co/settings/tokens

---

## Step 4: Build & Start GPUOpt + vLLM

```bash
docker compose -f docker-compose.local.yml up --build -d
```

This starts:
| Service | Port | Purpose |
|---|---|---|
| `api` | `8080` | GPUOpt backend (GPU monitoring + model management API) |
| `vllm` | `8001` | vLLM OpenAI-compatible server |

### Verify everything is running

```bash
# Check all containers are up
docker compose -f docker-compose.local.yml ps

# Check GPUOpt API health
curl http://localhost:8080/health/ready

# Check vLLM is serving
curl http://localhost:8001/v1/models
```

You should see the model you configured listed in the vLLM response.

---

## Step 5: Start the Frontend

### Option A: Using the built-in static frontend (no install)

If the React frontend is built (`frontend-react/dist/` exists), GPUOpt serves it automatically at `http://localhost:8080`.

To build the React frontend:

```bash
cd frontend-react
npm install
npm run build
```

Then restart the API container:

```bash
docker compose -f docker-compose.local.yml restart api
```

### Option B: Development mode (hot reload)

```bash
cd frontend-react
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Step 6: Use the SLM Inference Dashboard

1. Open the GPUOpt web UI (`http://localhost:8080` or `http://localhost:5173`)
2. Click **SLM Inference** in the sidebar
3. Select a model from the dropdown (e.g., `Qwen/Qwen2.5-3B-Instruct`)
4. Click **Start Model** — vLLM begins loading the model into GPU memory
5. Watch real-time GPU telemetry in the right panel:
   - GPU utilization %
   - Memory usage (bar + numeric)
   - Temperature
   - Power draw
   - Tokens/second (inferred from vLLM metrics)
6. Type a message in the chat box and hit Send
7. Click **Stop Model** to unload when done

---

## Step 7: GPU Monitoring Dashboard

The **Dashboard** and **GPU Status** pages show live telemetry from all GPUs:

- **Dashboard** — aggregate bar charts (utilization %, memory %, temperature)
- **GPU Status** — per-GPU cards with health color-coding
- **SLM Inference** — split view with chat + real-time GPU gauges

Telemetry is pushed via WebSocket every 5 seconds.

---

## Configuration Reference

### Changing the vLLM model

Edit `docker-compose.local.yml` and change the model argument:

```yaml
vllm:
  command: --model Qwen/Qwen2.5-3B-Instruct --port 8000
  # Change to:
  # command: --model microsoft/Phi-3-mini-4k-instruct --port 8000
```

Then recreate:

```bash
docker compose -f docker-compose.local.yml up -d --force-recreate vllm
```

### vLLM flags you can tune

| Flag | Default | Description |
|---|---|---|
| `--max-model-len` | auto | Max context length (e.g., `8192`) |
| `--gpu-memory-utilization` | `0.9` | Fraction of VRAM to use (e.g., `0.85`) |
| `--dtype` | `auto` | `auto`, `half`, `float16`, `bfloat16` |
| `--enforce-eager` | false | Disable CUDA graph optimization (saves VRAM) |
| `--max-num-seqs` | `256` | Max sequences to batch |
| `--quantization` | none | `awq`, `gptq`, `fp8` (reduces VRAM) |

Example with quantization:

```yaml
vllm:
  command: >
    --model Qwen/Qwen2.5-3B-Instruct
    --port 8000
    --quantization fp8
    --gpu-memory-utilization 0.85
```

### Changing GPUOpt telemetry polling interval

Set via environment variable in `docker-compose.local.yml`:

```yaml
api:
  environment:
    GPUOPT_GPU_POLL_INTERVAL: "5"   # default 15 seconds
```

---

## Troubleshooting

### "No NVIDIA GPU detected" in GPUOpt

```bash
# Verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check Docker Desktop GPU setting
# Settings → Resources → Advanced → show NVIDIA utilities
```

### vLLM container exits immediately

```bash
# View logs
docker compose -f docker-compose.local.yml logs vllm

# Common causes:
# - Out of VRAM → use a smaller model or add --gpu-memory-utilization 0.7
# - Missing Hugging Face token → add HUGGING_FACE_HUB_TOKEN to .env
# - Model name typo → verify model exists on huggingface.co
```

### Frontend shows "No GPU data"

- GPUOpt API container needs `--gpus all` access (already in compose file)
- Poll interval may be slow — wait 15 seconds for first snapshot
- Check API: `curl http://localhost:8080/api/v1/monitoring/gpu/snapshot`

### "CUDA out of memory" in vLLM

- Reduce `--gpu-memory-utilization` (e.g., `0.7`)
- Enable quantization: `--quantization fp8`
- Use a smaller model
- Set `--max-model-len 2048` to reduce KV cache

---

## Quick Commands Reference

```bash
# Start everything
docker compose -f docker-compose.local.yml up --build -d

# Stop everything
docker compose -f docker-compose.local.yml down

# View logs
docker compose -f docker-compose.local.yml logs -f api
docker compose -f docker-compose.local.yml logs -f vllm

# Restart a single service
docker compose -f docker-compose.local.yml restart vllm

# Rebuild and start
docker compose -f docker-compose.local.yml up --build -d

# Check GPU in vLLM container
docker compose -f docker-compose.local.yml exec vllm nvidia-smi
```

---

## Files Changed

| File | Change |
|---|---|
| `docker-compose.local.yml` | Added `vllm` service with GPU passthrough |
| `src/gpuopt/vllm_router.py` | New — vLLM management API (start/stop/status) |
| `src/gpuopt/main.py` | Register vLLM router |
| `frontend-react/src/components/SLMInference.tsx` | New — Inference panel with GPU telemetry |
| `frontend-react/src/App.tsx` | Added `/slm-inference` route |
| `frontend-react/src/components/Layout.tsx` | Added nav item |
| `frontend-react/src/services/api.ts` | Added vLLM API calls |

---

## Next Steps / Advanced

- **Add auto-scaling**: Watch GPU utilization and auto-stop vLLM when idle
- **Multi-model support**: Run vLLM with `--served-model-name` to alias models
- **Benchmarking**: Use the built-in `/api/v1/inference/benchmark` endpoint
- **Prometheus metrics**: vLLM exposes `/metrics` at port 8001 — scrape into GPUOpt's Prometheus
- **Slurm integration**: Route inference jobs via GPUOpt's Slurm connector
