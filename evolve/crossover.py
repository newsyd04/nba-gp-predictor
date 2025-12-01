import random
import copy
from tree.tree import Tree

def should_apply_crossover(crossover_rate=1.0) -> bool:
	return random.random() <= crossover_rate

def choose_parents(parents_pool: list[Tree]) -> tuple[Tree, Tree]:
	mrParent = random.choice(parents_pool)
	msParent = random.choice(parents_pool)
	return mrParent, msParent

def crossover(mrParent: Tree, msParent: Tree) -> tuple[Tree, Tree]:
    awesomeSon  = copy.deepcopy(mrParent)
    greatDaughter  = copy.deepcopy(msParent)

    node1 = awesomeSon.random_node()
    node2 = greatDaughter.random_node()

    node1.value, node2.value = node2.value, node1.value
    node1.left, node2.left = node2.left, node1.left
    node1.right, node2.right = node2.right, node1.right

    return awesomeSon, greatDaughter

def run_crossover(parents_pool: list[Tree], num_children: int, crossover_rate=1.0) -> list[Tree]:
    children = []

    while len(children) < num_children:
        mrParent, msParent = choose_parents(parents_pool)

        if should_apply_crossover(crossover_rate):
            awesomeSon, greatDaughter = crossover(mrParent, msParent)
        else:
            awesomeSon, greatDaughter = mrParent, msParent

        children.append(awesomeSon)
        if len(children) < num_children:
            children.append(greatDaughter)

    return children
