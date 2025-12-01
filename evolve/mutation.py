import random
import copy
from tree.tree import Tree

def should_mutate(mutation_rate=0.1) -> bool:
    return random.random() <= mutation_rate

def mutate(individual: Tree, max_depth=3) -> Tree:
    mutant = copy.deepcopy(individual)
    node = mutant.random_node()
    current_depth = node.current_depth
    max_depth = max_depth - current_depth if (max_depth - current_depth) > 1 else 1
    new_subtree = Tree()
    new_subtree.grow(max_depth, current_depth=current_depth)
    node.value = new_subtree.root.value
    node.left = new_subtree.root.left
    node.right = new_subtree.root.right
    return mutant

def run_mutations(children: list[Tree], max_depth: int, mutation_rate=0.1) -> list[Tree]:
    mutated = []
    for child in children:
        if should_mutate(mutation_rate):
            max_depth = max_depth - child.root.current_depth
            mutated.append(mutate(child, max_depth))
        else:
            mutated.append(child)
    return mutated
