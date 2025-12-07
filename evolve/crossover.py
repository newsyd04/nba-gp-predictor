import random
import copy
from tree.tree import Tree
from config import PARAMETERS

def should_apply_crossover(crossover_rate=1.0) -> bool:
    return random.random() <= crossover_rate

def choose_parents(parents_pool: list[Tree]) -> tuple[Tree, Tree]:
    mrParent = random.choice(parents_pool)
    msParent = random.choice(parents_pool)
    return mrParent, msParent

def crossover(mrParent: Tree, msParent: Tree) -> tuple[Tree, Tree]:
    awesomeSon = copy.deepcopy(mrParent)
    greatDaughter = copy.deepcopy(msParent)

    max_depth_allowed = PARAMETERS["max_tree_height"]

    node1 = awesomeSon.random_node()
    node2 = greatDaughter.random_node()

    parent1 = awesomeSon.find_parent(node1)
    parent2 = greatDaughter.find_parent(node2)

    old_node1 = copy.deepcopy(node1)
    old_node2 = copy.deepcopy(node2)

    if parent1 is None:
        awesomeSon.root = node2
    else:
        if parent1.left is node1:
            parent1.left = node2
        else:
            parent1.right = node2

    if parent2 is None:
        greatDaughter.root = node1
    else:
        if parent2.left is node2:
            parent2.left = node1
        else:
            parent2.right = node1

    if awesomeSon.depth() > max_depth_allowed:
        if parent1 is None:
            awesomeSon.root = old_node1
        else:
            if parent1.left is node2:
                parent1.left = old_node1
            else:
                parent1.right = old_node1

    if greatDaughter.depth() > max_depth_allowed:
        if parent2 is None:
            greatDaughter.root = old_node2
        else:
            if parent2.left is node1:
                parent2.left = old_node2
            else:
                parent2.right = old_node2

    return awesomeSon, greatDaughter

def run_crossover(parents_pool: list[Tree], num_children: int, crossover_rate=1.0) -> list[Tree]:
    children = []

    while len(children) < num_children:
        mrParent, msParent = choose_parents(parents_pool)

        if should_apply_crossover(crossover_rate):
            awesomeSon, greatDaughter = crossover(mrParent, msParent)
        else:
            awesomeSon = copy.deepcopy(mrParent)
            greatDaughter = copy.deepcopy(msParent)

        children.append(awesomeSon)
        if len(children) < num_children:
            children.append(greatDaughter)

    return children
