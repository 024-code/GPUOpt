from __future__ import annotations

import logging
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolicyGene:
    name: str
    action_type: str
    condition: str
    threshold: float
    weight: float = 1.0
    active: bool = True
    mutation_rate: float = 0.1


@dataclass
class PolicyChromosome:
    genes: List[PolicyGene]
    fitness_score: float = 0.0
    generation: int = 0

    def to_rego(self) -> str:
        rules = []
        for gene in self.genes:
            if gene.active:
                rule = f"""
allow {{
    input.action_type == "{gene.action_type}"
    {gene.condition}
    {gene.threshold}
    input.risk_score <= {gene.weight:.2f}
}}"""
                rules.append(rule)

        return """
package gpuopt.actions

default allow = false
""" + "\n".join(rules)


class PolicyEvolutionEngine:
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[PolicyChromosome] = []
        self.best_chromosome: Optional[PolicyChromosome] = None
        self.fitness_history: List[float] = []
        self.model_path = Path("models/policies/evolved_policies.pkl")

    def initialize_population(self) -> None:
        self.population = [self._create_random_chromosome() for _ in range(self.population_size)]

    def _create_random_chromosome(self) -> PolicyChromosome:
        genes = []
        action_types = ["scale_up", "scale_down", "cordon", "drain", "restart"]

        for action_type in action_types:
            conditions = [
                f"target_replicas <= {random.randint(1, 20)}",
                f"gpu_utilization >= {random.uniform(0.3, 0.8):.2f}",
                f"temperature <= {random.randint(60, 85)}",
                f"memory_usage <= {random.uniform(0.7, 0.95):.2f}",
                f"node_availability >= {random.uniform(0.5, 1.0):.2f}",
            ]
            num_conditions = random.randint(1, 2)
            selected_conditions = random.sample(conditions, num_conditions)
            condition = " && ".join(selected_conditions)

            gene = PolicyGene(
                name=f"rule_{action_type}_{len(genes)}",
                action_type=action_type,
                condition=condition,
                threshold=random.uniform(0.3, 0.9),
                weight=random.uniform(0.5, 1.0),
                active=random.choice([True, False]),
            )
            genes.append(gene)

        return PolicyChromosome(genes=genes)

    def evaluate_fitness(self, chromosome: PolicyChromosome, metrics: Dict) -> float:
        efficiency = metrics.get("gpu_utilization", 0.5) * 0.6
        success_rate = 1 - metrics.get("failure_rate", 0.1)
        success_score = success_rate * 0.3
        temp_score = max(0.0, 1 - metrics.get("temperature", 50) / 100) * 0.1
        complexity = len(chromosome.genes)
        complexity_bonus = min(1.0, complexity / 10) * 0.05
        active_ratio = sum(1 for g in chromosome.genes if g.active) / max(len(chromosome.genes), 1)
        activity_score = active_ratio * 0.05

        fitness = efficiency + success_score + temp_score + complexity_bonus + activity_score

        if active_ratio < 0.2:
            fitness *= 0.5

        chromosome.fitness_score = fitness
        return fitness

    def select_parents(self) -> Tuple[PolicyChromosome, PolicyChromosome]:
        weights = [c.fitness_score for c in self.population]
        min_weight = min(weights) if weights else 0
        weights = [w - min_weight + 0.1 for w in weights]

        if sum(weights) == 0:
            weights = [1.0] * len(self.population)

        parents = random.choices(self.population, weights=weights, k=2)
        return parents[0], parents[1]

    def crossover(
        self, parent1: PolicyChromosome, parent2: PolicyChromosome
    ) -> Tuple[PolicyChromosome, PolicyChromosome]:
        if random.random() > self.crossover_rate:
            return parent1, parent2

        crossover_point = random.randint(1, len(parent1.genes) - 1)

        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        child1 = PolicyChromosome(genes=child1_genes, generation=parent1.generation + 1)
        child2 = PolicyChromosome(genes=child2_genes, generation=parent2.generation + 1)

        return child1, child2

    def mutate(self, chromosome: PolicyChromosome) -> PolicyChromosome:
        for gene in chromosome.genes:
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(["threshold", "weight", "condition", "toggle"])

                if mutation_type == "threshold":
                    gene.threshold = random.uniform(0.3, 0.9)
                elif mutation_type == "weight":
                    gene.weight = random.uniform(0.5, 1.0)
                elif mutation_type == "condition":
                    gene.condition = re.sub(
                        r"\d+\.?\d*",
                        lambda m: str(float(m.group(0)) * random.uniform(0.8, 1.2)),
                        gene.condition,
                    )
                elif mutation_type == "toggle":
                    gene.active = not gene.active

        return chromosome

    def evolve(
        self, metrics_history: List[Dict], generations: int | None = None
    ) -> PolicyChromosome:
        if generations is None:
            generations = self.generations

        if not self.population:
            self.initialize_population()

        best_fitness = -float("inf")
        best_chromosome: Optional[PolicyChromosome] = None

        for generation in range(generations):
            for chromosome in self.population:
                if metrics_history:
                    keys = metrics_history[0].keys()
                    avg_metrics = {
                        k: float(np.mean([m.get(k, 0) for m in metrics_history[-100:]]))
                        for k in keys
                    }
                else:
                    avg_metrics = {
                        "gpu_utilization": 0.6,
                        "failure_rate": 0.1,
                        "temperature": 50,
                    }

                fitness = self.evaluate_fitness(chromosome, avg_metrics)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_chromosome = chromosome

            new_population: List[PolicyChromosome] = []

            sorted_population = sorted(
                self.population, key=lambda x: x.fitness_score, reverse=True
            )
            new_population.extend(sorted_population[:2])

            while len(new_population) < self.population_size:
                p1, p2 = self.select_parents()
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.extend([c1, c2])

            self.population = new_population[: self.population_size]
            self.fitness_history.append(best_fitness)

            if generation % 10 == 0:
                logger.info(
                    "Generation %d, Best Fitness: %.4f", generation, best_fitness
                )

        if best_chromosome is not None:
            self.best_chromosome = best_chromosome

        return self.best_chromosome  # type: ignore[return-value]

    def get_best_policy(self) -> str:
        if self.best_chromosome is not None:
            return self.best_chromosome.to_rego()
        return ""

    def save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "best_chromosome": self.best_chromosome,
                    "population": self.population,
                    "fitness_history": self.fitness_history,
                },
                f,
            )
        logger.info("Evolved policies saved to %s", self.model_path)

    def load_model(self) -> None:
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.best_chromosome = data.get("best_chromosome")
                self.population = data.get("population", [])
                self.fitness_history = data.get("fitness_history", [])
            logger.info("Evolved policies loaded from %s", self.model_path)
        else:
            logger.info("No existing evolved policies found, starting fresh")
