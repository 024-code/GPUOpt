<#
.SYNOPSIS
    GPUOpt Complete Setup Script for Windows / PowerShell
.DESCRIPTION
    Sets up the GPUOpt backend sandbox environment: GPU detection, venv, 
    dependencies, configuration, cluster seeding, tests, and API server.
.PARAMETER SkipGPUCheck
    Skip NVIDIA GPU detection via nvidia-smi
.PARAMETER SkipAPIServer
    Setup without starting the API server
.PARAMETER Interactive
    Open interactive menu mode
.PARAMETER Command
    Run a specific command: api, test, seed, plan, bench, swagger, kind-up, kind-down, gpu, help
.PARAMETER Help
    Show this help message
.EXAMPLE
    .\setup-gpuopt.ps1
    Full setup + API server (port 8080)
.EXAMPLE
    .\setup-gpuopt.ps1 -Interactive
    Interactive menu mode
.EXAMPLE
    .\setup-gpuopt.ps1 -Command api
    Start API server only
#>
param(
    [switch]$SkipGPUCheck,
    [switch]$SkipAPIServer,
    [switch]$Interactive,
    [string]$Command,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path | Out-String | Write-Host
    exit 0
}

# ------------------------------------------------------------
# COLORS
# ------------------------------------------------------------
function Write-Success { Write-Host "[OK] $($args[0])" -ForegroundColor Green }
function Write-Error   { Write-Host "[FAIL] $($args[0])" -ForegroundColor Red }
function Write-Info    { Write-Host "[..] $($args[0])" -ForegroundColor Cyan }
function Write-Step    { Write-Host "[>>] $($args[0])" -ForegroundColor Yellow }
function Write-Command { Write-Host "     $($args[0])" -ForegroundColor Gray }

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
function Ensure-Venv {
    if (-not (Test-Path ".venv")) {
        Write-Step "Creating virtual environment..."
        python -m venv .venv
    }
    $activate = Join-Path $PWD ".venv/Scripts/Activate.ps1"
    . $activate
    if (-not $env:VIRTUAL_ENV) {
        Write-Error "Failed to activate virtual environment."
        Write-Info "Try: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
        exit 1
    }
    Write-Success "Virtual environment active: $env:VIRTUAL_ENV"
}

function Ensure-API-Running {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health/ready" -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -eq 200)
    } catch { return $false }
}

function Wait-For-API {
    Write-Step "Waiting for API server to be ready..."
    for ($i = 1; $i -le 30; $i++) {
        if (Ensure-API-Running) { Write-Success "API server is ready!"; return $true }
        Write-Command "Attempt $i/30..."
        Start-Sleep -Seconds 1
    }
    Write-Error "API server did not become ready in time."
    return $false
}

function Ensure-ProjectRoot {
    if (-not (Test-Path "pyproject.toml")) {
        Write-Error "Not in project root. Run from the directory containing pyproject.toml"
        exit 1
    }
}

function Set-PythonPath {
    $env:PYTHONPATH = (Resolve-Path "src").Path
}

# ------------------------------------------------------------
# 1. GPU CHECK
# ------------------------------------------------------------
function Test-GPU {
    Write-Step "Checking NVIDIA GPU..."
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "NVIDIA GPU not found or nvidia-smi not available."
        return $false
    }
    Write-Success "GPU Detected: $gpuInfo"

    Write-Step "Testing PyTorch GPU detection..."
    $code = @'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    x = torch.randn(4096, 4096).cuda()
    y = torch.randn(4096, 4096).cuda()
    z = torch.matmul(x, y)
    print("[OK] GPU test successful!")
else:
    print("[WARN] CUDA not available. Running on CPU.")
'@
    $tmp = [System.IO.Path]::GetTempFileName() + ".py"
    Set-Content -Path $tmp -Value $code -Encoding utf8
    python $tmp
    Remove-Item $tmp -ErrorAction SilentlyContinue
    return $true
}

# ------------------------------------------------------------
# 2. INSTALL DEPENDENCIES
# ------------------------------------------------------------
function Install-Dependencies {
    Write-Step "Installing dependencies..."
    python -m pip install --upgrade pip -q

    Write-Command "Installing GPUOpt project..."
    pip install -e ".[dev]" -q

    Write-Command "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

    Write-Command "Installing AI/ML packages..."
    pip install transformers accelerate bitsandbytes sentencepiece -q

    Write-Success "All dependencies installed."
}

# ------------------------------------------------------------
# 3. CONFIGURE ENVIRONMENT
# ------------------------------------------------------------
function Configure-Environment {
    Write-Step "Configuring environment..."
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item .env.example .env
            Write-Success ".env created from .env.example"
        } else {
            @"
GPUOPT_ENV=development
GPUOPT_DATABASE_PATH=./data/gpuopt.db
GPUOPT_LOG_LEVEL=INFO
GPUOPT_ALLOW_MOCK_GPU=true
GPUOPT_CHECK_TIMEOUT_SECONDS=15
GPUOPT_API_HOST=0.0.0.0
GPUOPT_API_PORT=8080
GPUOPT_CORS_ORIGINS=["*"]
GPUOPT_API_KEY=
GPUOPT_API_KEY_HEADER=X-API-Key
GPUOPT_RATE_LIMIT_PER_MINUTE=120
GPUOPT_RATE_LIMIT_PER_HOUR=5000
"@ | Out-File -FilePath .env -Encoding utf8
            Write-Success ".env created with defaults"
        }
    } else {
        Write-Success ".env already exists"
    }
}

# ------------------------------------------------------------
# 4. SEED MOCK CLUSTER
# ------------------------------------------------------------
function Seed-Cluster {
    Write-Step "Seeding mock cluster..."
    if (Test-Path "environments.mock.yaml") {
        Set-PythonPath
        python -m gpuopt.cli seed --file environments.mock.yaml
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Mock cluster seeded successfully."
        } else {
            Write-Error "Seed command failed."
        }
    } else {
        Write-Error "environments.mock.yaml not found."
    }
}

# ------------------------------------------------------------
# 5. RUN UNIT TESTS
# ------------------------------------------------------------
function Run-Tests {
    Write-Step "Running unit tests ($(python -c "import gpuopt; print(gpuopt.__version__)" 2>$null))..."
    pytest tests/ -v --tb=short
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All unit tests passed."
    } else {
        Write-Error "Some tests failed."
    }
}

# ------------------------------------------------------------
# 6. START API SERVER
# ------------------------------------------------------------
function Start-API {
    Write-Step "Starting GPUOpt API server..."
    Write-Command "Server: http://127.0.0.1:8080"
    Write-Command "Docs:   http://127.0.0.1:8080/docs"
    Write-Command "Press Ctrl+C to stop."
    Write-Command ""
    Set-PythonPath
    uvicorn gpuopt.main:app --reload --host 0.0.0.0 --port 8080
}

# ------------------------------------------------------------
# 7. TEST INFERENCE PLANNER
# ------------------------------------------------------------
function Test-Planner {
    Write-Step "Testing inference planner..."
    if (-not (Ensure-API-Running)) {
        Write-Error "API server is not running. Start it with: .\setup-gpuopt.ps1 -Command api"
        return
    }
    $payload = Get-Content "examples/plan-8b.json" -Raw -Encoding utf8
    $tmp = [System.IO.Path]::GetTempFileName()
    Set-Content -Path $tmp -Value $payload -Encoding utf8
    Write-Command "POST /api/v1/inference/plan (llama-8b, fp16, 4K context)..."
    $response = curl.exe -s -X POST "http://127.0.0.1:8080/api/v1/inference/plan" -H "Content-Type: application/json" -d "@${tmp}"
    $response | ConvertFrom-Json | ConvertTo-Json
    Write-Success "Planner test completed."
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------
# 8. RUN BENCHMARK
# ------------------------------------------------------------
function Run-Benchmark {
    Write-Step "Running inference benchmark..."
    if (-not (Ensure-API-Running)) {
        Write-Error "API server is not running. Start it with: .\setup-gpuopt.ps1 -Command api"
        return
    }
    $payload = Get-Content "examples/benchmark-real.json" -Raw -Encoding utf8
    $tmp = [System.IO.Path]::GetTempFileName()
    Set-Content -Path $tmp -Value $payload -Encoding utf8
    Write-Command "POST /api/v1/inference/benchmark (20 requests, concurrency 4)..."
    $response = curl.exe -s -X POST "http://127.0.0.1:8080/api/v1/inference/benchmark" -H "Content-Type: application/json" -d "@${tmp}"
    $response | ConvertFrom-Json | ConvertTo-Json
    Write-Success "Benchmark completed."
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------
# 9. OPEN SWAGGER
# ------------------------------------------------------------
function Open-Swagger {
    Write-Step "Opening Swagger UI..."
    Start-Process "http://127.0.0.1:8080/docs"
    Write-Success "Swagger UI opened in browser."
}

# ------------------------------------------------------------
# 10. KIND CLUSTER MANAGEMENT
# ------------------------------------------------------------
function Kind-Up {
    Write-Step "Deploying kind cluster..."
    $kindVer = & kind version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "kind is not installed. Install with: winget install kind"
        return
    }
    Write-Command "kind version: $kindVer"

    Write-Command "Creating cluster 'gpuopt'..."
    kind create cluster --name gpuopt --config infra/kind/kind-config.yaml

    Write-Command "Labeling worker nodes with mock GPUs..."
    $workers = & kubectl get nodes --context kind-gpuopt -l 'node-role.kubernetes.io/control-plane!=true' -o name 2>$null
    if ($LASTEXITCODE -eq 0) {
        foreach ($n in $workers) {
            $name = $n -replace '^node/'
            kubectl label nodes --context kind-gpuopt $name gpuopt.ai/mock-gpu-count=2 --overwrite
        }
    }

    Write-Command "Building and loading GPUOpt image..."
    docker build -t gpuopt-backend:latest .
    kind load docker-image gpuopt-backend:latest --name gpuopt

    Write-Command "Deploying GPUOpt manifests..."
    kubectl create namespace gpuopt-system --context kind-gpuopt 2>$null
    kubectl apply -f infra/k8s/base/ --context kind-gpuopt 2>$null
    kubectl apply -f infra/k8s/mock-dcgm/ --context kind-gpuopt 2>$null

    Write-Command "Waiting for deployment..."
    kubectl wait --context kind-gpuopt -n gpuopt-system deployment/gpuopt-backend --for=condition=Available --timeout=120s 2>$null

    Write-Command "Port-forwarding (Ctrl+C to stop)..."
    kubectl port-forward --context kind-gpuopt -n gpuopt-system service/gpuopt-backend 8080:8080
}

function Kind-Down {
    Write-Step "Destroying kind cluster..."
    kind delete cluster --name gpuopt
    Write-Success "Kind cluster destroyed."
}

# ------------------------------------------------------------
# 11. INTERACTIVE MENU
# ------------------------------------------------------------
function Show-Menu {
    Clear-Host
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "   GPUOpt Control Center" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] Full Setup (Install + Test + API)" -ForegroundColor White
    Write-Host "  [2] Start API Server" -ForegroundColor White
    Write-Host "  [3] Run Unit Tests" -ForegroundColor White
    Write-Host "  [4] Seed Mock Cluster" -ForegroundColor White
    Write-Host "  [5] Test Inference Planner" -ForegroundColor White
    Write-Host "  [6] Run Benchmark" -ForegroundColor White
    Write-Host "  [7] Open Swagger UI" -ForegroundColor White
    Write-Host "  [8] Test GPU" -ForegroundColor White
    Write-Host "  [9] Deploy Kind Cluster" -ForegroundColor White
    Write-Host "  [0] Destroy Kind Cluster" -ForegroundColor White
    Write-Host "  [h] Help" -ForegroundColor White
    Write-Host "  [q] Quit" -ForegroundColor White
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "API Status: " -NoNewline
    if (Ensure-API-Running) { Write-Host "[RUNNING]" -ForegroundColor Green }
    else { Write-Host "[STOPPED]" -ForegroundColor Red }
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
    $choice = Read-Host "Select option"
    return $choice
}

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
Ensure-ProjectRoot

if ($Command) {
    Ensure-Venv
    switch ($Command) {
        "api" { Start-API }
        "test" { Run-Tests }
        "seed" { Seed-Cluster }
        "plan" { Test-Planner }
        "bench" { Run-Benchmark }
        "swagger" { Open-Swagger }
        "gpu" { Test-GPU }
        "kind-up" { Kind-Up }
        "kind-down" { Kind-Down }
        "help" { Get-Content $MyInvocation.MyCommand.Path | Select-String -Pattern "^#" | ForEach-Object { $_ -replace "^#\s?", "" } }
        default { Write-Error "Unknown command: $Command" }
    }
    exit 0
}

if ($Interactive) {
    Ensure-Venv
    while ($true) {
        $choice = Show-Menu
        switch ($choice) {
            "1" {
                Write-Step "Running full setup..."
                if (-not $SkipGPUCheck) { Test-GPU }
                Install-Dependencies
                Configure-Environment
                Seed-Cluster
                Run-Tests
                Start-API
            }
            "2" { Start-API }
            "3" { Run-Tests }
            "4" { Seed-Cluster }
            "5" { Test-Planner }
            "6" { Run-Benchmark }
            "7" { Open-Swagger }
            "8" { Test-GPU }
            "9" { Kind-Up }
            "0" { Kind-Down }
            "h" { Get-Content $MyInvocation.MyCommand.Path | Select-String -Pattern "^#" | ForEach-Object { $_ -replace "^#\s?", "" } }
            "q" { Write-Host "Goodbye!" -ForegroundColor Cyan; exit 0 }
            default { Write-Error "Invalid option" }
        }
        Read-Host "`nPress Enter to continue..."
    }
    exit 0
}

# ------------------------------------------------------------
# DEFAULT: Full non-interactive setup
# ------------------------------------------------------------
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   GPUOpt Setup Script" -ForegroundColor Cyan
Write-Host "   RTX 4090 | Local Development | Inference" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

Ensure-Venv

if (-not $SkipGPUCheck) { Test-GPU }

Install-Dependencies
Configure-Environment
Seed-Cluster
Run-Tests

if (-not $SkipAPIServer) {
    Start-API
} else {
    Write-Info "API server skipped. Start manually with:"
    Write-Command ".\setup-gpuopt.ps1 -Command api"
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "[OK] GPUOpt setup complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick Commands:" -ForegroundColor White
Write-Host "   .\setup-gpuopt.ps1 -Interactive       - Interactive menu" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command api        - Start API server" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command test       - Run unit tests" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command plan       - Test planner" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command bench      - Run benchmark" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command swagger    - Open Swagger UI" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command kind-up    - Deploy kind cluster" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command kind-down  - Destroy kind cluster" -ForegroundColor Gray
Write-Host "   .\setup-gpuopt.ps1 -Command gpu        - Test GPU" -ForegroundColor Gray
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor White
Write-Host "   Planner:  curl -X POST http://127.0.0.1:8080/api/v1/inference/plan -H `"Content-Type: application/json`" --data-binary @examples/plan-8b.json" -ForegroundColor Gray
Write-Host "   Benchmark: curl -X POST http://127.0.0.1:8080/api/v1/inference/benchmark -H `"Content-Type: application/json`" --data-binary @examples/benchmark-real.json" -ForegroundColor Gray
Write-Host "   Summary:   curl http://127.0.0.1:8080/api/v1/environments/summary" -ForegroundColor Gray
Write-Host "   Swagger:   http://127.0.0.1:8080/docs" -ForegroundColor Gray
