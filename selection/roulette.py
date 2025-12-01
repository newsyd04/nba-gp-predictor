import random
from selection.selection_interface import ISelection
from tree.tree import Tree

class RouletteSelection(ISelection):
    def create_parents_pool(self, population: list[Tree], fitness_scores: list[float]) -> list[Tree]:
        if len(population) != len(fitness_scores):
            raise ValueError("Population and fitness scores must have the same length.")

        if not population:
            return []

        parents_pool = []
        wheel = self._make_wheel(population, fitness_scores)
        for _ in population:
            parent = self._select_one(wheel)
            parents_pool.append(parent)
        return parents_pool

    def _make_wheel(self, population: list[Tree], fitness_scores: list[float]) -> list[tuple[Tree, float]]:
        total = sum(fitness_scores)

        if total == 0:
            prob = 1 / len(population)
            return [(individual, prob) for individual in population]

        wheel = [(individual, fitness/total) for individual, fitness in zip(population, fitness_scores)]

        return wheel

    def _select_one(self, wheel: list[tuple[Tree, float]]) -> Tree:
        r = random.random()
        cumulative = 0.0
        for individual, prob in wheel:
            cumulative += prob
            if r <= cumulative:
                return individual

        return wheel[-1][0]
