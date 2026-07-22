from __future__ import annotations

import pytest

from gpuopt.registry import ServiceRegistry, get_registry, reset_registry


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


class TestServiceRegistry:
    def test_register_and_get(self):
        reg = ServiceRegistry()
        reg.register("scheduler", {"type": "rl"})
        assert reg.get("scheduler") == {"type": "rl"}

    def test_register_duplicate_raises(self):
        reg = ServiceRegistry()
        reg.register("scheduler", "v1")
        with pytest.raises(KeyError, match="already registered"):
            reg.register("scheduler", "v2")

    def test_register_duplicate_force(self):
        reg = ServiceRegistry()
        reg.register("scheduler", "v1")
        reg.register("scheduler", "v2", force=True)
        assert reg.get("scheduler") == "v2"

    def test_get_default(self):
        reg = ServiceRegistry()
        assert reg.get("nonexistent") is None
        assert reg.get("nonexistent", "default") == "default"

    def test_get_or_create_creates(self):
        reg = ServiceRegistry()
        result = reg.get_or_create("counter", int, 42)
        assert result == 42
        assert reg.get("counter") == 42

    def test_get_or_create_reuses(self):
        reg = ServiceRegistry()
        reg.register("obj", {"key": "value"})
        result = reg.get_or_create("obj", dict, key="other")
        assert result == {"key": "value"}

    def test_remove(self):
        reg = ServiceRegistry()
        reg.register("test", "value")
        reg.remove("test")
        assert reg.get("test") is None

    def test_list(self):
        reg = ServiceRegistry()
        reg.register("a", 1)
        reg.register("b", 2)
        assert reg.list() == {"a": 1, "b": 2}

    def test_clear(self):
        reg = ServiceRegistry()
        reg.register("a", 1)
        reg.clear()
        assert reg.list() == {}


class TestGlobalRegistry:
    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_registry(self):
        reg = get_registry()
        reg.register("test", "value")
        reset_registry()
        assert reg.get("test") is None


class TestRouterRegistryIntegration:
    def test_scheduler_router_creates_default(self):
        from gpuopt.scheduler.router import _get_scheduler
        from gpuopt.scheduler.rl_scheduler import RLScheduler
        sched = _get_scheduler()
        assert isinstance(sched, RLScheduler)

    def test_predictor_router_creates_default(self):
        from gpuopt.predictor.router import _get_predictor
        from gpuopt.predictor.failure_predictor import FailurePredictor
        pred = _get_predictor()
        assert isinstance(pred, FailurePredictor)

    def test_policy_router_creates_default(self):
        from gpuopt.policy.router import _get_engine
        from gpuopt.policy.evolution import PolicyEvolutionEngine
        engine = _get_engine()
        assert isinstance(engine, PolicyEvolutionEngine)

    def test_healing_router_creates_default(self):
        from gpuopt.healing.router import _get_healer
        from gpuopt.healing.auto_healer import AutoHealer
        healer = _get_healer()
        assert isinstance(healer, AutoHealer)

    def test_services_are_singletons(self):
        from gpuopt.scheduler.router import _get_scheduler
        from gpuopt.predictor.router import _get_predictor
        from gpuopt.policy.router import _get_engine
        from gpuopt.healing.router import _get_healer
        assert _get_scheduler() is _get_scheduler()
        assert _get_predictor() is _get_predictor()
        assert _get_engine() is _get_engine()
        assert _get_healer() is _get_healer()
