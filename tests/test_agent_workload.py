from __future__ import annotations

from src.gpuopt.schemas import WorkloadInput
from src.gpuopt.workload_agent import (
    DigitalTwinSimulator,
    MLPredictor,
    SystemDetector,
    WorkloadAgent,
)
from src.gpuopt.workload_agent_router import agent


def test_system_detector_runs() -> None:
    detector = SystemDetector()
    info = detector.detect()
    assert info.hostname != ""
    assert info.cpu_cores >= 0
    assert info.gpu_count >= 0
    assert info.detected_at != ""


def test_ml_predictor_returns_result() -> None:
    predictor = MLPredictor()
    wl = WorkloadInput(name="test-job", gpu_required=1, memory_required_gb=8)
    system = SystemDetector().detect()
    result = predictor.predict(wl, system)
    assert 0.0 <= result.success_probability <= 1.0
    assert result.predicted_duration_minutes >= 0
    assert isinstance(result.risk_factors, list)
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation) > 0


def test_ml_predictor_training() -> None:
    predictor = MLPredictor()
    wl = WorkloadInput(name="train-job", gpu_required=1, memory_required_gb=4)
    system = SystemDetector().detect()
    result1 = predictor.predict(wl, system)
    r = predictor.train([
        {"workload": wl.model_dump(mode="json"), "actual_success": True, "actual_duration": 10.0},
        {"workload": wl.model_dump(mode="json"), "actual_success": True, "actual_duration": 12.0},
    ])
    assert r["samples"] == 2
    result2 = predictor.predict(wl, system)
    assert result2.success_probability >= 0


def test_ml_predictor_risk_factors() -> None:
    predictor = MLPredictor()
    wl = WorkloadInput(name="big-job", gpu_required=999, memory_required_gb=99999)
    system = SystemDetector().detect()
    result = predictor.predict(wl, system)
    assert len(result.risk_factors) > 0
    assert result.success_probability < 0.5


def test_digital_twin_simulator() -> None:
    sim = DigitalTwinSimulator()
    wl = WorkloadInput(name="sim-job", gpu_required=1, memory_required_gb=1)
    system = SystemDetector().detect()
    prediction = MLPredictor().predict(wl, system)
    result = sim.simulate(wl, system, prediction)
    assert result.simulation_id != ""
    assert result.workload.name == "sim-job"
    assert isinstance(result.feasible, bool)
    assert isinstance(result.assigned_gpu_indices, list)
    if result.feasible:
        assert len(result.assigned_gpu_indices) > 0
        assert result.assigned_memory_gb > 0
        assert result.estimated_cost > 0


def test_digital_twin_simulator_no_gpu() -> None:
    sim = DigitalTwinSimulator()
    wl = WorkloadInput(name="no-gpu-job", gpu_required=999, memory_required_gb=1)
    system = SystemDetector().detect()
    prediction = MLPredictor().predict(wl, system)
    result = sim.simulate(wl, system, prediction)
    assert not result.feasible
    assert len(result.rejection_reason) > 0


def test_workload_agent_submit() -> None:
    a = WorkloadAgent()
    wl = WorkloadInput(name="agent-job", gpu_required=1, memory_required_gb=2)
    result = a.submit_workload(wl)
    assert "status" in result
    assert "workload" in result
    assert "system" in result
    assert "prediction" in result
    if result["status"] == "assigned":
        assert "assignment" in result
        assert "assignment_id" in result["assignment"]


def test_workload_agent_full_lifecycle() -> None:
    a = WorkloadAgent()
    wl = WorkloadInput(name="lifecycle-job", gpu_required=1, memory_required_gb=1)
    result = a.submit_workload(wl)
    if result["status"] == "assigned":
        aid = result["assignment"]["assignment_id"]
        assert a.get_assignment(aid) is not None
        assignments = a.list_assignments()
        assert any(ass.assignment_id == aid for ass in assignments)
        completed = a.complete_assignment(aid, success=True, duration_minutes=15.0)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.actual_success is True
        assert completed.actual_duration_minutes == 15.0
        stats = a.get_stats()
        assert stats["total_assignments"] >= 1


def test_workload_agent_stats() -> None:
    a = WorkloadAgent()
    stats = a.get_stats()
    assert "total_assignments" in stats
    assert "assigned" in stats
    assert "completed" in stats
    assert "failed" in stats
    assert "success_rate" in stats
    assert "ml_training_samples" in stats


def test_workload_agent_router_initialized() -> None:
    assert agent is not None
    assert hasattr(agent, "detect_system")
    assert hasattr(agent, "submit_workload")
    assert hasattr(agent, "list_assignments")


def test_system_detector_detailed() -> None:
    detector = SystemDetector()
    info = detector.detect("test-cluster")
    assert info.cluster_id == "test-cluster"
    assert isinstance(info.ram_total_gb, float)
    assert isinstance(info.ram_available_gb, float)
    assert isinstance(info.ram_used_gb, float)
    assert isinstance(info.ram_usage_percent, float)
    assert isinstance(info.gpu_count, int)
    assert isinstance(info.gpus, list)


def test_ml_predictor_with_training_data_shift() -> None:
    predictor = MLPredictor()
    wl = WorkloadInput(name="shift-job", gpu_required=1, memory_required_gb=4)
    system = SystemDetector().detect()
    before = predictor.predict(wl, system)
    predictor.train([
        {"workload": wl.model_dump(mode="json"), "actual_success": False, "actual_duration": 5.0},
        {"workload": wl.model_dump(mode="json"), "actual_success": False, "actual_duration": 6.0},
        {"workload": wl.model_dump(mode="json"), "actual_success": False, "actual_duration": 7.0},
        {"workload": wl.model_dump(mode="json"), "actual_success": False, "actual_duration": 8.0},
    ])
    after = predictor.predict(wl, system)
    assert before.success_probability >= 0
    assert after.success_probability >= 0


def test_digital_twin_simulator_estimated_cost() -> None:
    sim = DigitalTwinSimulator()
    wl = WorkloadInput(name="cost-job", gpu_required=2, memory_required_gb=8)
    system = SystemDetector().detect()
    prediction = MLPredictor().predict(wl, system)
    result = sim.simulate(wl, system, prediction)
    if result.feasible:
        assert result.estimated_cost > 0
        assert len(result.assigned_gpu_indices) <= 2


def test_complete_nonexistent_assignment() -> None:
    a = WorkloadAgent()
    result = a.complete_assignment("nonexistent-id", success=False)
    assert result is None


def test_get_nonexistent_assignment() -> None:
    a = WorkloadAgent()
    result = a.get_assignment("nonexistent-id")
    assert result is None
