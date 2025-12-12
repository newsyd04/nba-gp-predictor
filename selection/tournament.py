import random
from selection.selection_interface import ISelection
from tree.tree import Tree

# Implements tournament selection for genetic programming
class TournamentSelection(ISelection):
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    # Creates a parents pool using tournament selection
    def create_parents_pool(self, population: list[Tree], fitness_scores: list[float]) -> list[Tree]:
        if len(population) != len(fitness_scores):
            raise ValueError("Population and fitness scores must have the same length.")

        if not population:
            return []

        parents_pool = []
        pop_with_fitness = list(zip(population, fitness_scores))

        for _ in population:
            parent = self._select_one(pop_with_fitness)
            parents_pool.append(parent)

        return parents_pool
    
    # Tournament selection: choose k at random, pick the best.
    def _select_one(self, pop_with_fitness: list[tuple[Tree, float]]) -> Tree:
        """
        Tournament selection: choose k at random, pick the best.
        """
        tournament = random.sample(pop_with_fitness, self.tournament_size)
        winner, _ = max(tournament, key=lambda x: x[1])
        return winner