import random
import copy
from tree.tree import Tree

def should_mutate(mutation_rate=0.1) -> bool:
    return random.random() <= mutation_rate

def mutate(individual: Tree, max_depth=3) -> Tree:
    """
    Improved mutation:
    - 70% chance: small local mutation (operator / constant / variable)
    - 30% chance: subtree replacement
    """
    mutant = copy.deepcopy(individual)

    # Choose a random node
    node = mutant.random_node()

    # --- LEAF MUTATIONS (micro-mutation) ---
    if node.is_leaf():
        r = random.random()

        # 40% chance: tweak constant slightly
        if isinstance(node.value, (int, float)) and r < 0.40:
            node.value += round(random.uniform(-1.5, 1.5), 2)
            return mutant

        # 40% chance: switch variable
        if isinstance(node.value, str) and r < 0.80:
            node.value = random.choice(individual.variables)
            return mutant

        # 20% chance: replace leaf entirely
        node.value = (
            random.choice(individual.variables)
            if random.random() < 0.5
            else round(random.uniform(-10, 10), 2)
        )
        return mutant

    # --- INTERNAL NODE MUTATIONS (micro-mutation) ---
    # 40%: replace operator only
    if random.random() < 0.40:
        from config import OPERATIONS
        node.value = random.choice(OPERATIONS)
        return mutant

    # --- STRUCTURAL MUTATION (macro-mutation) ---
    # Replace subtree, but avoid giant random trees
    # Keep it small and controlled
    allowed_remaining = max_depth - node.current_depth
    if allowed_remaining < 1:
        allowed_remaining = 1

    new_subtree = Tree(
        variables=individual.variables,
        current_depth=node.current_depth
    )

    new_subtree.grow(
        max_depth=node.current_depth + allowed_remaining,
        current_depth=node.current_depth
    )

    # Replace entire subtree
    node.value = new_subtree.root.value
    node.left = new_subtree.root.left
    node.right = new_subtree.root.right

    return mutant

# Runs mutations on a list of children based on mutation rate
def run_mutations(children: list[Tree], max_depth: int, mutation_rate=0.1) -> list[Tree]:
    mutated = []

    for child in children:
        if should_mutate(mutation_rate):
            mutated.append(mutate(child, max_depth))
        else:
            mutated.append(child)

    return mutated
