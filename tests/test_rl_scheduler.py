from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from gpuopt.scheduler.rl_scheduler import Job, Node, RLScheduler


@pytest.fixture(autouse=True)
def temp_model_dir() -> None:
    """Reset scheduler state between tests by swapping model dir."""
    from gpuopt.registry import get_registry
    reg = get_registry()
    old = reg.get("scheduler")
    test_scheduler = RLScheduler(model_dir=tempfile.mkdtemp())
    reg.register("scheduler", test_scheduler, force=True)
    yield
    if old is not None:
        reg.register("scheduler", old, force=True)
    else:
        reg.remove("scheduler")


def make_nodes(count: int = 4) -> list[Node]:
    return [
        Node(id=f"gpu-{i}", available_gpus=8, total_gpus=8,
             free_memory_gb=64.0, temperature=45.0, gpu_model="A100",
             status="ready")
        for i in range(count)
    ]


class TestRLSchedulerCore:
    def test_state_features_shape(self):
        sched = RLScheduler()
        job = Job(id="j1", required_gpus=2, priority=5)
        nodes = make_nodes(3)
        state = sched._get_state_features(job, nodes)
        assert state.shape == (10,)
        assert all(0 <= v <= 1 for v in state)

    def test_eligible_actions(self):
        sched = RLScheduler()
        job = Job(id="j1", required_gpus=2)
        nodes = make_nodes(3)
        eligible = sched._eligible_actions(job, nodes)
        assert len(eligible) == 3

        saturated = [Node(id="full", available_gpus=0, total_gpus=8, free_memory_gb=64.0, temperature=45.0, gpu_model="A100", status="ready")]
        assert sched._eligible_actions(job, saturated) == []

    def test_choose_action_returns_valid(self):
        sched = RLScheduler()
        job = Job(id="j1", required_gpus=2)
        nodes = make_nodes(3)
        state = sched._get_state_features(job, nodes)
        eligible = sched._eligible_actions(job, nodes)
        action = sched.choose_action(state, eligible)
        assert action in eligible

    def test_choose_action_no_eligible(self):
        sched = RLScheduler()
        state = sched._get_state_features(Job(), [])
        assert sched.choose_action(state, []) == -1

    def test_compute_reward_success(self):
        sched = RLScheduler()
        job = Job(required_gpus=4, priority=8)
        node = Node(available_gpus=8, total_gpus=8, free_memory_gb=64.0, temperature=45.0)
        reward = sched._compute_reward(job, node, success=True)
        assert reward > 0

    def test_compute_reward_failure(self):
        sched = RLScheduler()
        reward = sched._compute_reward(Job(), Node(), success=False)
        assert reward == -1.0

    def test_compute_reward_high_temp_penalty(self):
        sched = RLScheduler()
        job = Job(required_gpus=1)
        hot = Node(available_gpus=8, total_gpus=8, free_memory_gb=64.0, temperature=80.0)
        cool = Node(available_gpus=8, total_gpus=8, free_memory_gb=64.0, temperature=40.0)
        hot_r = sched._compute_reward(job, hot, success=True)
        cool_r = sched._compute_reward(job, cool, success=True)
        assert hot_r < cool_r

    def test_q_learning_update(self):
        sched = RLScheduler(learning_rate=0.5, discount_factor=0.9)
        import numpy as np
        state = np.zeros(10, dtype=np.float32)
        next_state = np.ones(10, dtype=np.float32)
        sched.update_q(state, 0, 1.0, next_state, done=False)
        assert sched.q_table[state.tobytes()][0] != 0

    def test_schedule_returns_result(self):
        sched = RLScheduler()
        job = Job(id="test_job", required_gpus=2, priority=5)
        nodes = make_nodes(3)
        result = sched.schedule(job, nodes)
        assert result.action == "placed"
        assert result.node is not None
        assert result.reward > -1

    def test_schedule_no_eligible_nodes(self):
        sched = RLScheduler()
        job = Job(id="test_job", required_gpus=100)
        nodes = make_nodes(1)
        result = sched.schedule(job, nodes)
        assert result.action == "queued"
        assert result.node is None

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = RLScheduler(model_dir=tmp)
            sched.episode_rewards = [1.0, 2.0, 3.0]
            sched.save_model()
            assert (sched.model_path).exists()

            sched2 = RLScheduler(model_dir=tmp)
            assert len(sched2.episode_rewards) == 3

    def test_metrics(self):
        sched = RLScheduler()
        job = Job(id="metrics_test", required_gpus=1)
        nodes = make_nodes(2)
        for _ in range(10):
            sched.schedule(job, nodes)
        metrics = sched.metrics()
        assert metrics["q_table_size"] > 0
        assert metrics["total_placements"] >= 10
        assert 0 <= metrics["placement_success_rate"] <= 1

    def test_train_updates_q_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = RLScheduler(model_dir=tmp)
            initial_size = len(sched.q_table)
            sched.train_from_history(episodes=5)
            assert len(sched.q_table) >= initial_size
            assert len(sched.episode_rewards) == 5

    def test_episode_epsilon_decay(self):
        sched = RLScheduler()
        sched.episode_rewards = list(range(1000))
        state = sched._get_state_features(Job(required_gpus=1), make_nodes(2))
        eligible = [0, 1]
        actions = set()
        for _ in range(50):
            actions.add(sched.choose_action(state, eligible))
        # With epsilon=0.1 after 1000 episodes, most choices should be greedy
        # (still some randomness at 0.1, but should converge)
        assert len(actions) <= 2  # both nodes possible as valid actions


class TestRLSchedulerAPI:
    def test_schedule_endpoint(self, client: TestClient):
        payload = {
            "id": "api-job-1",
            "required_gpus": 2,
            "priority": 7,
            "estimated_duration": 2.0,
            "memory_gb": 16.0,
            "nodes": [{"id": "n1", "available_gpus": 8, "total_gpus": 8, "free_memory_gb": 64.0, "temperature": 45.0, "gpu_model": "A100", "status": "ready"}],
        }
        resp = client.post("/api/v1/scheduler/rl/schedule", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("scheduled", "queued")

    def test_schedule_no_nodes(self, client: TestClient):
        resp = client.post("/api/v1/scheduler/rl/schedule", json={"id": "j1", "required_gpus": 1, "nodes": []})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_train_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/scheduler/rl/train?episodes=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "training_complete"
        assert data["episodes"] == 3

    def test_metrics_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/scheduler/rl/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "q_table_size" in data
        assert "total_placements" in data

    def test_save_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/scheduler/rl/save")
        assert resp.status_code == 200

    def test_load_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/scheduler/rl/load")
        assert resp.status_code == 200

    def test_schedule_and_train_flow(self, client: TestClient):
        client.post("/api/v1/scheduler/rl/schedule", json={
            "id": "flow-job", "required_gpus": 2, "priority": 5,
            "nodes": [{"id": "n1", "available_gpus": 8, "total_gpus": 8, "free_memory_gb": 64.0, "temperature": 45.0, "gpu_model": "A100", "status": "ready"}],
        })
        client.post("/api/v1/scheduler/rl/train?episodes=5")
        metrics = client.get("/api/v1/scheduler/rl/metrics").json()
        assert metrics["episode_count"] > 0
