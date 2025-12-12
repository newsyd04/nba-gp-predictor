import math
import time
from tree.tree import Tree
from tree.tree_node import TreeNode

# Evaluates the fitness of an individual tree
def fitness_worker(args: tuple[Tree, list[int], list[list[float]], list[str]]) -> tuple[float, float, float]:
    individual, train_target_outputs, train_data, variables = args
    return fitness(individual, train_target_outputs, train_data, variables)

# Calculates fitness, prediction time, and loss time for an individual
def fitness(individual: Tree, target_outputs: list[int], data: list[list[float]], variables: list[str]) -> tuple[float, float, float]:
    t_predictions_start = time.time()
    predictions = get_predictions(individual, data, variables)
    t_predictions_end = time.time()
    t_predictions_duration = t_predictions_end - t_predictions_start

    t_loss_start = time.time()
    loss = cross_entropy_loss(predictions, target_outputs)
    t_loss_end = time.time()
    t_loss_duration = t_loss_end - t_loss_start

    # tree size penalty
    size = individual.size()

    return _fitness(loss, t_predictions_duration, t_loss_duration, size)

# Generates predictions for the given data using the provided tree
def get_predictions(tree: Tree, data: list[list[float]], variables: list[str]) -> list[float]:
    predictions = []
    var_index = {var: i for i, var in enumerate(variables)}
    for row in data:
        raw_output = evaluate_tree(tree.root, row, var_index)

        # clamp + scale before squashing
        raw_output = max(min(raw_output, 10), -10)
        raw_output /= 10

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
    # smooth squashing to (0, 1)
    return (math.tanh(x) + 1) / 2

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

# Calculates the fitness value with a penalty for tree size
def _fitness(loss: float, prediction_time: float, loss_time: float, size: int) -> tuple[float, float, float]:
    """
    Base fitness from loss, then penalise large trees.
    fitness = exp(-loss) * 1 / (1 + alpha * size)
    """
    base = math.exp(-loss)

    # bloat penalty
    alpha = 0.0005
    penalty = 1 / (1 + alpha * (size ** 0.7))

    fitness_value = base * penalty
    return fitness_value, prediction_time, loss_time

# Computes the average fitness of the population
def average_fitness(fitness_scores: list[float], population: list[Tree]) -> float:
    return sum(fitness_scores) / len(population)

# Finds the maximum fitness and corresponding individual in the population
def max_fitness(fitness_scores: list[float], population: list[Tree]) -> tuple[float, Tree | None]:
    max_fit, max_individual = float('-inf'), None
    for score, individual in zip(fitness_scores, population):
        if score > max_fit:
            max_fit = score
            max_individual = individual
    return max_fit, max_individual

# Returns the population sorted by fitness in descending order
def get_population_sorted_by_fitness(population: list[Tree], fitness_scores: list[float]) -> list[tuple[Tree, float]]:
    paired = list(zip(population, fitness_scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired

# Recursively evaluates the tree for a given data row
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

    op = node.value

    # Handle unary operations
    if op in ["abs(x)", "log(|x| + 1)", "tanh", "relu"]:
        x = evaluate_tree(node.left, row, var_index)

        try:
            if op == "abs(x)":
                return abs(x)
            elif op == "log(|x| + 1)":
                return math.log(abs(x) + 1)
            elif op == "tanh":
                return math.tanh(x)
            elif op == "relu":
                return x if x > 0 else 0
        except:
            return 0

    # Handle binary operations   
    left_value = evaluate_tree(node.left, row, var_index)
    right_value = evaluate_tree(node.right, row, var_index)

    try:
        if op == '+':
            return left_value + right_value
        elif op == '-':
            return left_value - right_value
        elif op == '*':
            return left_value * right_value
        elif op == '/':
            return left_value / right_value if right_value != 0 else 1  # avoid continuous division by zero
        else:
            raise ValueError(f"Unknown operator: {op}")

    except Exception:
        return 0
