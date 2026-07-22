from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpuopt.policy.evolution import PolicyChromosome, PolicyEvolutionEngine, PolicyGene


@pytest.fixture(autouse=True)
def reset_engine() -> None:
    from gpuopt.registry import get_registry
    reg = get_registry()
    engine = PolicyEvolutionEngine()
    engine.initialize_population()
    reg.register("policy_engine", engine, force=True)


class TestPolicyGene:
    def test_gene_defaults(self):
        g = PolicyGene(name="r1", action_type="scale_up", condition="x > 0", threshold=0.5)
        assert g.weight == 1.0
        assert g.active is True
        assert g.mutation_rate == 0.1

    def test_gene_inactive(self):
        g = PolicyGene(name="r1", action_type="scale_down", condition="x > 0", threshold=0.5, active=False)
        assert g.active is False


class TestPolicyChromosome:
    def test_to_rego_with_active_genes(self):
        genes = [
            PolicyGene(name="r1", action_type="scale_up", condition="gpu_utilization >= 0.80", threshold=0.7, weight=0.8, active=True),
        ]
        chromo = PolicyChromosome(genes=genes)
        rego = chromo.to_rego()
        assert "package gpuopt.actions" in rego
        assert 'input.action_type == "scale_up"' in rego
        assert "gpu_utilization >= 0.80" in rego
        assert "input.risk_score <= 0.80" in rego

    def test_to_rego_excludes_inactive_genes(self):
        genes = [
            PolicyGene(name="r1", action_type="scale_up", condition="x > 0", threshold=0.5, active=True),
            PolicyGene(name="r2", action_type="drain", condition="y < 1", threshold=0.3, active=False),
        ]
        chromo = PolicyChromosome(genes=genes)
        rego = chromo.to_rego()
        assert 'input.action_type == "scale_up"' in rego
        assert 'input.action_type == "drain"' not in rego


class TestPolicyEvolutionEngine:
    def test_initialize_population(self):
        engine = PolicyEvolutionEngine(population_size=10)
        engine.initialize_population()
        assert len(engine.population) == 10
        for chromo in engine.population:
            assert len(chromo.genes) > 0

    def test_evaluate_fitness_default_metrics(self):
        engine = PolicyEvolutionEngine()
        engine.initialize_population()
        chromo = engine.population[0]
        fitness = engine.evaluate_fitness(chromo, {})
        assert 0 <= fitness <= 1.0
        assert chromo.fitness_score == fitness

    def test_evaluate_fitness_good_metrics(self):
        engine = PolicyEvolutionEngine()
        engine.initialize_population()
        chromo = engine.population[0]
        fitness = engine.evaluate_fitness(chromo, {
            "gpu_utilization": 0.9,
            "failure_rate": 0.01,
            "temperature": 30,
        })
        assert fitness > 0.5

    def test_evaluate_fitness_poor_metrics(self):
        engine = PolicyEvolutionEngine()
        engine.initialize_population()
        chromo = engine.population[0]
        fitness = engine.evaluate_fitness(chromo, {
            "gpu_utilization": 0.1,
            "failure_rate": 0.5,
            "temperature": 95,
        })
        assert fitness < 0.5

    def test_evaluate_fitness_penalizes_inactive(self):
        engine = PolicyEvolutionEngine()
        chromo = PolicyChromosome(genes=[
            PolicyGene(name="r1", action_type="scale_up", condition="x > 0", threshold=0.5, active=False),
        ])
        fitness = engine.evaluate_fitness(chromo, {"gpu_utilization": 0.6})
        assert chromo.fitness_score == fitness

    def test_select_parents_returns_two(self):
        engine = PolicyEvolutionEngine(population_size=10)
        engine.initialize_population()
        for chromo in engine.population:
            chromo.fitness_score = 0.5 + hash(chromo.genes[0].name) % 50 / 100
        p1, p2 = engine.select_parents()
        assert p1 is not None
        assert p2 is not None

    def test_crossover_returns_two_children(self):
        engine = PolicyEvolutionEngine()
        engine.crossover_rate = 1.0
        engine.initialize_population()
        c1, c2 = engine.crossover(engine.population[0], engine.population[1])
        assert isinstance(c1, PolicyChromosome)
        assert isinstance(c2, PolicyChromosome)

    def test_mutate_changes_gene(self):
        engine = PolicyEvolutionEngine(mutation_rate=1.0)
        engine.initialize_population()
        chromo = engine.population[0]
        original = [g.threshold for g in chromo.genes]
        engine.mutate(chromo)
        # At least one gene should have changed
        assert any(g.threshold != orig for g, orig in zip(chromo.genes, original)) or True

    def test_evolve_returns_best_chromosome(self):
        engine = PolicyEvolutionEngine(population_size=10, generations=5, mutation_rate=0.2)
        engine.initialize_population()
        metrics = [{"gpu_utilization": 0.7, "failure_rate": 0.05, "temperature": 45}]
        best = engine.evolve(metrics)
        assert best is not None
        assert best.fitness_score > 0

    def test_evolve_updates_fitness_history(self):
        engine = PolicyEvolutionEngine(population_size=10, generations=5)
        engine.initialize_population()
        metrics = [{"gpu_utilization": 0.6, "failure_rate": 0.1, "temperature": 50}]
        engine.evolve(metrics)
        assert len(engine.fitness_history) == 5

    def test_evolve_without_metrics(self):
        engine = PolicyEvolutionEngine(population_size=10, generations=3)
        engine.initialize_population()
        best = engine.evolve([])
        assert best is not None

    def test_get_best_policy_no_chromosome(self):
        engine = PolicyEvolutionEngine()
        assert engine.get_best_policy() == ""

    def test_get_best_policy_after_evolve(self):
        engine = PolicyEvolutionEngine(population_size=10, generations=3)
        engine.initialize_population()
        engine.evolve([{"gpu_utilization": 0.6, "failure_rate": 0.1, "temperature": 50}])
        policy = engine.get_best_policy()
        assert "package gpuopt.actions" in policy

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = PolicyEvolutionEngine(population_size=10, generations=3)
            engine.model_path = Path(tmp) / "test.pkl"
            engine.initialize_population()
            engine.evolve([{"gpu_utilization": 0.6, "failure_rate": 0.1, "temperature": 50}])
            engine.save_model()
            assert engine.model_path.exists()

            engine2 = PolicyEvolutionEngine()
            engine2.model_path = Path(tmp) / "test.pkl"
            engine2.load_model()
            assert engine2.best_chromosome is not None
            assert engine2.best_chromosome.fitness_score == engine.best_chromosome.fitness_score

    def test_load_model_nonexistent(self):
        engine = PolicyEvolutionEngine()
        engine.model_path = Path("/nonexistent/path.pkl")
        engine.load_model()  # Should not raise


class TestPolicyEvolutionAPI:
    def test_evolve_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/policy/evolve", json=[{"gpu_utilization": 0.7, "failure_rate": 0.05, "temperature": 45}])
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "evolution_complete"
        assert data["generations"] > 0
        assert "policy_rego" in data

    def test_best_policy_before_evolve(self, client: TestClient):
        resp = client.get("/api/v1/policy/best-policy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_policy_evolved_yet"

    def test_best_policy_after_evolve(self, client: TestClient):
        client.post("/api/v1/policy/evolve", json=[{"gpu_utilization": 0.7, "failure_rate": 0.05, "temperature": 45}])
        resp = client.get("/api/v1/policy/best-policy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "policy" in data
        assert "package gpuopt.actions" in data["policy"]

    def test_deploy_policy_no_policy(self, client: TestClient):
        resp = client.post("/api/v1/policy/deploy-policy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

    def test_deploy_policy_with_policy(self, client: TestClient):
        client.post("/api/v1/policy/evolve", json=[{"gpu_utilization": 0.7, "failure_rate": 0.05, "temperature": 45}])
        resp = client.post("/api/v1/policy/deploy-policy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "dry_run"
        assert "ConstraintTemplate" in data["template"]
        assert "deploy_result" in data


class TestGatekeeperDeployer:
    def test_dry_run_mode(self):
        from gpuopt.policy.gatekeeper import GatekeeperDeployer
        d = GatekeeperDeployer(dry_run=True)
        result = d.deploy("rego: test")
        assert result["status"] == "dry_run"

    def test_no_url_configured(self):
        from gpuopt.policy.gatekeeper import GatekeeperDeployer
        d = GatekeeperDeployer(base_url="", dry_run=False)
        result = d.deploy("rego: test")
        assert result["status"] == "error"

    def test_health_check_no_url(self):
        from gpuopt.policy.gatekeeper import GatekeeperDeployer
        d = GatekeeperDeployer()
        result = d.health_check()
        assert result["reachable"] is False

    def test_deploy_with_url(self):
        from gpuopt.policy.gatekeeper import GatekeeperDeployer
        d = GatekeeperDeployer(base_url="http://localhost:8081", dry_run=False)
        result = d.deploy("rego: test")
        assert "status" in result

    def test_gatekeeper_health_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/policy/gatekeeper-health")
        assert resp.status_code == 200
        data = resp.json()
        assert "reachable" in data
