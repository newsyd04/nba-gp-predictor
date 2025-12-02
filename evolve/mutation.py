import random
import copy
from tree.tree import Tree

def should_mutate(mutation_rate=0.1) -> bool:
    return random.random() <= mutation_rate


def mutate(individual: Tree, max_depth=3) -> Tree:
    """
    Replace a random subtree with a newly grown subtree using the SAME VARIABLES
    as the original tree.
    """
    mutant = copy.deepcopy(individual)

    node = mutant.random_node()
    current_depth = node.current_depth

    # ensure subtree stays within allowed depth
    allowed_remaining = max_depth - current_depth
    if allowed_remaining < 1:
        allowed_remaining = 1

    # pass variables into new subtree
    new_subtree = Tree(
        variables=individual.variables,
        current_depth=current_depth
    )

    new_subtree.grow(
        max_depth=current_depth + allowed_remaining,
        current_depth=current_depth
    )

    # structurally replace node
    node.value = new_subtree.root.value
    node.left = new_subtree.root.left
    node.right = new_subtree.root.right

    return mutant


def run_mutations(children: list[Tree], max_depth: int, mutation_rate=0.1) -> list[Tree]:
    """
    Apply mutation independently to each child.
    """
    mutated = []

    for child in children:
        if should_mutate(mutation_rate):
            mutated.append(mutate(child, max_depth))
        else:
            mutated.append(child)

    return mutated
