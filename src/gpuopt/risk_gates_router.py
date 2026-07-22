from __future__ import annotations

from fastapi import APIRouter

from gpuopt.risk_gates import RiskGatesService
from gpuopt.risk_gates_schemas import (
    AutoscaleConfig,
    AutoscaleGateInput,
    AutoscaleGateResult,
    GateAction,
    GateResult,
    K8sMutationGateInput,
    K8sMutationGateResult,
    MemoryBenchmarkConfig,
    MemoryGateInput,
    MemoryGateResult,
    MockDataGateInput,
    MockDataGateResult,
    MoeExpertBalanceInput,
    MoeGateResult,
    QualityGateInput,
    QualityGateResult,
    QualityTestConfig,
    RiskGatesDashboard,
    SecretScanInput,
    SecretScanResult,
    TpBenchmarkConfig,
    TpGateInput,
    TpGateResult,
)

router = APIRouter(prefix="/api/v1/risk-gates", tags=["risk_gates"])
_gates = RiskGatesService()


@router.get("/dashboard", response_model=RiskGatesDashboard)
def get_dashboard() -> RiskGatesDashboard:
    return _gates.evaluate_all()


@router.post("/memory", response_model=MemoryGateResult)
def evaluate_memory(input_data: MemoryGateInput) -> MemoryGateResult:
    return _gates.memory.evaluate(input_data)


@router.post("/mock-data", response_model=MockDataGateResult)
def evaluate_mock_data(input_data: MockDataGateInput) -> MockDataGateResult:
    return _gates.mock_data.evaluate(input_data)


@router.post("/tp-communication", response_model=TpGateResult)
def evaluate_tp(input_data: TpGateInput) -> TpGateResult:
    return _gates.tp_comm.evaluate(input_data)


@router.post("/autoscale", response_model=AutoscaleGateResult)
def evaluate_autoscale(input_data: AutoscaleGateInput) -> AutoscaleGateResult:
    return _gates.autoscale.evaluate(input_data)


@router.post("/quality", response_model=QualityGateResult)
def evaluate_quality(input_data: QualityGateInput) -> QualityGateResult:
    return _gates.quality.evaluate(input_data)


@router.post("/k8s-mutation", response_model=K8sMutationGateResult)
def evaluate_k8s(input_data: K8sMutationGateInput) -> K8sMutationGateResult:
    return _gates.k8s_mutation.evaluate(input_data)


@router.post("/secrets", response_model=SecretScanResult)
def evaluate_secrets(input_data: SecretScanInput) -> SecretScanResult:
    return _gates.secrets.evaluate(input_data)


@router.post("/moe-imbalance", response_model=MoeGateResult)
def evaluate_moe(input_data: MoeExpertBalanceInput) -> MoeGateResult:
    return _gates.moe.evaluate(input_data)


@router.get("/health")
def health() -> dict:
    return _gates.health()
