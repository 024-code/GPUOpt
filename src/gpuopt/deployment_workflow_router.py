from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gpuopt.deployment_workflow import DeploymentWorkflowService
from gpuopt.deployment_workflow_schemas import (
    Step1Input,
    Step1Output,
    Step2Input,
    Step2Output,
    Step3Input,
    Step3Output,
    Step4Input,
    Step4Output,
    Step5Input,
    Step5Output,
    Step6Input,
    Step6Output,
    Step7Input,
    Step7Output,
    WorkflowState,
)

router = APIRouter(prefix="/api/v1/deployment-workflow", tags=["deployment_workflow"])
_service = DeploymentWorkflowService()


# ── Workflow Lifecycle ────────────────────────────────────────


@router.get("/health")
def health() -> dict:
    return _service.health()


@router.get("/", response_model=list[WorkflowState])
def list_workflows() -> list[WorkflowState]:
    return _service.list_workflows()


@router.get("/{workflow_id}", response_model=WorkflowState)
def get_workflow(workflow_id: str) -> WorkflowState:
    wf = _service.get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    return wf


@router.delete("/{workflow_id}")
def delete_workflow(workflow_id: str) -> dict:
    if not _service.delete_workflow(workflow_id):
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    return {"status": "deleted", "workflow_id": workflow_id}


@router.get("/{workflow_id}/next-step")
def get_next_step(workflow_id: str) -> dict:
    return _service.get_next_step(workflow_id)


# ── Step 1: Model Identity ────────────────────────────────────


@router.post("/step1", response_model=dict)
def run_step1(input_data: Step1Input, workflow_id: str | None = None) -> dict:
    wf, output = _service.step1(input_data, workflow_id)
    return {"workflow_id": wf.workflow_id, "current_step": 1, "output": output.model_dump()}


# ── Step 2: Hardware Specification ────────────────────────────


@router.post("/{workflow_id}/step2", response_model=dict)
def run_step2(workflow_id: str, input_data: Step2Input) -> dict:
    wf, output = _service.step2(input_data, workflow_id)
    return {"workflow_id": wf.workflow_id, "current_step": 2, "output": output.model_dump()}


# ── Step 3: SLO Requirements ──────────────────────────────────


@router.post("/{workflow_id}/step3", response_model=dict)
def run_step3(workflow_id: str, input_data: Step3Input) -> dict:
    wf, output = _service.step3(input_data, workflow_id)
    return {"workflow_id": wf.workflow_id, "current_step": 3, "output": output.model_dump()}


# ── Step 4: Deployment ────────────────────────────────────────


@router.post("/{workflow_id}/step4", response_model=dict)
def run_step4(workflow_id: str, input_data: Step4Input) -> dict:
    wf, output = _service.step4(input_data, workflow_id)
    return {"workflow_id": wf.workflow_id, "current_step": 4, "output": output.model_dump()}


# ── Step 5: Benchmark ─────────────────────────────────────────


@router.post("/{workflow_id}/step5", response_model=dict)
def run_step5(workflow_id: str, input_data: Step5Input) -> dict:
    wf, output = _service.step5(input_data, workflow_id)
    return {"workflow_id": wf.workflow_id, "current_step": 5, "output": output.model_dump()}


# ── Step 6: Production Replica Count ──────────────────────────


@router.post("/{workflow_id}/step6", response_model=dict)
def run_step6(workflow_id: str, input_data: Step6Input) -> dict:
    wf, output = _service.step6(input_data, workflow_id)
    return {"workflow_id": wf.workflow_id, "current_step": 6, "output": output.model_dump()}


# ── Step 7: Optimization Experiments ──────────────────────────


@router.post("/{workflow_id}/step7", response_model=dict)
def run_step7(workflow_id: str, input_data: Step7Input) -> dict:
    wf, output = _service.step7(input_data, workflow_id)
    return {"workflow_id": wf.workflow_id, "current_step": 7, "output": output.model_dump(), "summary": output.optimization_summary}
