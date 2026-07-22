from __future__ import annotations

import logging
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str = ""
    required_gpus: int = 1
    priority: int = 5
    estimated_duration: float = 1.0
    memory_gb: float = 8.0
    checkpointable: bool = False


@dataclass
class Node:
    id: str = ""
    available_gpus: int = 8
    total_gpus: int = 8
    free_memory_gb: float = 64.0
    temperature: float = 45.0
    gpu_model: str = "A100"
    status: str = "ready"


@dataclass
class PlacementResult:
    node: Node | None = None
    action: str = "queued"
    node_id: str = ""
    reward: float = 0.0
    success: bool = False
    reasoning: str = ""
    q_value: float = 0.0


class RLScheduler:
    def __init__(
        self,
        state_size: int = 10,
        action_size: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        model_dir: str = "models",
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.q_table: defaultdict[bytes, np.ndarray] = defaultdict(lambda: np.zeros(action_size))
        self.experience_buffer: list[dict[str, Any]] = []
        self.max_buffer_size = 10000
        self.episode_rewards: list[float] = []
        self.placement_history: list[dict[str, Any]] = []
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "rl_scheduler.pkl"
        self.load_model()

    # ── State representation ──────────────────────────────────

    def _get_state_features(self, job: Job, nodes: list[Node]) -> np.ndarray:
        available_gpus = [n.available_gpus for n in nodes]
        temperatures = [n.temperature for n in nodes]
        memory_usage = [n.free_memory_gb for n in nodes]
        max_gpus = max((n.total_gpus for n in nodes), default=1)

        features = [
            min(job.required_gpus / max_gpus, 1.0),
            job.priority / 10.0,
            min(job.estimated_duration / 24.0, 1.0),
            np.mean(available_gpus) / max_gpus if max_gpus > 0 else 0,
            min(available_gpus) / max_gpus if max_gpus > 0 and available_gpus else 0,
            max(available_gpus) / max_gpus if max_gpus > 0 and available_gpus else 0,
            1 - (sum(available_gpus) / (len(nodes) * max_gpus)) if nodes and max_gpus > 0 else 0,
            np.mean(temperatures) / 100.0 if temperatures else 0,
            len([n for n in nodes if n.available_gpus >= job.required_gpus]) / max(len(nodes), 1),
            np.mean(memory_usage) / 100.0 if memory_usage else 0,
        ]
        return np.array(features, dtype=np.float32)

    def _eligible_actions(self, job: Job, nodes: list[Node]) -> list[int]:
        return [
            i for i, n in enumerate(nodes)
            if n.available_gpus >= job.required_gpus and n.free_memory_gb > 0
        ]

    def _compute_reward(self, job: Job, node: Node, success: bool) -> float:
        if not success:
            return -1.0
        reward = 1.0
        reward += (job.required_gpus / max(node.total_gpus, 1)) * 0.5
        if node.temperature > 75:
            reward -= 0.3
        if node.temperature > 85:
            reward -= 0.5
        reward += (job.priority / 10.0) * 0.2
        mem_eff = job.memory_gb / max(node.free_memory_gb + job.memory_gb, 1)
        reward += (1 - mem_eff) * 0.2
        return reward

    # ── Core RL loop ──────────────────────────────────────────

    def choose_action(self, state: np.ndarray, eligible: list[int]) -> int:
        if not eligible:
            return -1
        epsilon = max(0.1, 1.0 - len(self.episode_rewards) / 1000)
        if random.random() < epsilon:
            return random.choice(eligible)
        q_values = self.q_table[state.tobytes()]
        return max(eligible, key=lambda x: q_values[x])

    def update_q(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        key = state.tobytes()
        nkey = next_state.tobytes()
        current = self.q_table[key][action]
        future = np.max(self.q_table[nkey]) if not done else 0
        self.q_table[key][action] = current + self.lr * (reward + self.gamma * future - current)
        self.experience_buffer.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def schedule(self, job: Job, nodes: list[Node]) -> PlacementResult:
        state = self._get_state_features(job, nodes)
        eligible = self._eligible_actions(job, nodes)

        if not eligible:
            return PlacementResult(action="queued", reasoning="No eligible nodes available")

        action = self.choose_action(state, eligible)
        if action < 0 or action >= len(nodes):
            return PlacementResult(action="queued", reasoning="No valid action found")

        node = nodes[action]
        q_val = float(self.q_table[state.tobytes()][action])
        success = random.random() > 0.05
        reward = self._compute_reward(job, node, success)
        next_nodes = self._simulate(nodes, node, job, success)
        next_state = self._get_state_features(job, next_nodes)
        self.update_q(state, action, reward, next_state, not success)

        self.placement_history.append({"job_id": job.id, "node_id": node.id, "reward": reward, "success": success})

        return PlacementResult(
            node=node, action="placed", node_id=node.id,
            reward=reward, success=success,
            q_value=q_val,
            reasoning=self._explain(q_val, node),
        )

    def train_from_history(self, episodes: int = 100) -> None:
        for ep in range(episodes):
            total = 0.0
            for _ in range(20):
                job = Job(
                    id=f"train_job_{random.randint(1, 10000)}",
                    required_gpus=random.randint(1, 4),
                    priority=random.randint(1, 10),
                    estimated_duration=random.uniform(0.5, 12),
                    memory_gb=random.uniform(4, 16),
                    checkpointable=random.choice([True, False]),
                )
                nodes = [
                    Node(
                        id=f"node_{i}",
                        available_gpus=random.randint(1, 8), total_gpus=8,
                        free_memory_gb=random.uniform(8, 64),
                        temperature=random.uniform(30, 85),
                        gpu_model="A100", status="ready",
                    )
                    for i in range(3)
                ]
                result = self.schedule(job, nodes)
                total += result.reward
            self.episode_rewards.append(total)
            if ep % 10 == 0 or ep == episodes - 1:
                avg = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                logger.info("Training episode %d/%d, avg reward: %.2f", ep + 1, episodes, avg)
        self.save_model()

    # ── Persistence ───────────────────────────────────────────

    def save_model(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "episode_rewards": self.episode_rewards,
            "placement_history": self.placement_history[-1000:],
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("RL model saved to %s (q-table size=%d)", self.model_path, len(self.q_table))

    def load_model(self) -> None:
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(self.action_size), data.get("q_table", {}))
                self.episode_rewards = data.get("episode_rewards", [])
                self.placement_history = data.get("placement_history", [])
            logger.info("RL model loaded from %s (q-table size=%d)", self.model_path, len(self.q_table))
        else:
            logger.info("No existing RL model at %s, starting fresh", self.model_path)

    def metrics(self) -> dict[str, Any]:
        recent = self.placement_history[-1000:]
        success_rate = sum(1 for p in recent if p["success"]) / max(len(recent), 1)
        return {
            "q_table_size": len(self.q_table),
            "experience_buffer_size": len(self.experience_buffer),
            "episode_count": len(self.episode_rewards),
            "average_reward": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
            "placement_success_rate": round(success_rate, 4),
            "total_placements": len(self.placement_history),
        }

    # ── Internal helpers ──────────────────────────────────────

    def _simulate(self, nodes: list[Node], selected: Node, job: Job, success: bool) -> list[Node]:
        if not success:
            return [Node(**n.__dict__) for n in nodes]
        updated: list[Node] = []
        for n in nodes:
            copy = Node(**n.__dict__)
            if copy.id == selected.id:
                copy.available_gpus = max(0, copy.available_gpus - job.required_gpus)
                copy.free_memory_gb = max(0, copy.free_memory_gb - job.memory_gb)
                copy.temperature = min(100, copy.temperature + 2)
            else:
                copy.temperature = max(30, copy.temperature - 0.5)
            updated.append(copy)
        return updated

    @staticmethod
    def _explain(q_val: float, node: Node) -> str:
        if q_val > 0.8:
            return f"High confidence on {node.id} (Q={q_val:.2f})"
        if q_val > 0.5:
            return f"Moderate confidence on {node.id} (Q={q_val:.2f})"
        return f"Exploring placement on {node.id} (Q={q_val:.2f})"
