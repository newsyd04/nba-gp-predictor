import math
from tree.tree import Tree
from tree.tree_node import TreeNode

def fitness(individual: Tree, target_outputs: list[int], data: list[list[float]], variables: list[str]) -> float:
    predictions = get_predictions(individual, data, variables)
    loss = cross_entropy_loss(predictions, target_outputs)
    return _fitness(loss)

def get_predictions(tree: Tree, data: list[list[float]], variables: list[str]) -> list[float]:
    predictions = []
    var_index = {var: i for i, var in enumerate(variables)}
    for row in data:
        raw_output = evaluate_tree(tree.root, row, var_index)
        probability = sigmoid(raw_output)
        predictions.append(probability)
    return predictions

def sigmoid(x: float) -> float:
    # prevents overflow for large positive x
    if x > 50:
        return 1.0
    # prevents underflow for large negative x
    if x < -50:
        return 0.0
    return 1 / (1 + math.exp(-x))

# -1/N summation(y*log(p) + (1-y)*log(1-p))
def cross_entropy_loss(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("Length of predictions and targets must be the same.")
    # adding a small number to prevent log(0)
    eps = 1e-12
    total = 0.0
    for p, y in zip(predictions, targets):
        p = min(max(p, eps), 1 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(predictions)

def _fitness(loss: float) -> float:
    return 1 / (1 + loss)

def total_fitness(population: list[Tree], target_outputs: list[float], data: list[list[float]], variables: list[str]) -> float:
    total = 0.0
    for individual in population:
        total += fitness(individual, target_outputs, data, variables)
    return total

def average_fitness(population: list[Tree], target_outputs: list[float], data: list[list[float]], variables: list[str]) -> float:
    return total_fitness(population, target_outputs, data, variables) / len(population)

def max_fitness(population: list[Tree], target_outputs: list[float], data: list[list[float]], variables: list[str]) -> tuple[float, Tree | None]:
    max_fit, max_individual = float('-inf'), None
    for individual in population:
        fit = fitness(individual, target_outputs, data, variables)
        if fit > max_fit:
            max_fit = fit
            max_individual = individual
    return max_fit, max_individual

def evaluate_tree(node: TreeNode, row: list[float], var_index: dict[str, int]) -> float:
    if node is None:
        return 0

    if node.is_leaf():
        value = node.value
        if isinstance(value, str):
            index = var_index.get(value, None)
            return row[index] if index is not None else 0
        else:
            return value

    left_value = evaluate_tree(node.left, row, var_index)
    right_value = evaluate_tree(node.right, row, var_index)

    op = node.value
    try:
        if op == '+':
            return left_value + right_value
        elif op == '-':
            return left_value - right_value
        elif op == '*':
            return left_value * right_value
        elif op == '/':
            return left_value / right_value if right_value != 0 else 1 # avoid continuous division by zero
        else:
            raise ValueError(f"Unknown operator: {op}")

    except Exception:
        return 0
