from tree.tree import Tree

def make_population(size: int, max_tree_height: int) -> list:
    population = []

    base = size // max_tree_height
    remainder = size % max_tree_height

    heights = []
    # max_tree_height down to 1 and adds the remainder to the tallest trees
    for height in range(max_tree_height, 0, -1):
        count = base + (1 if remainder > 0 else 0)
        heights.extend([height] * count)
        if remainder > 0:
            remainder -= 1

    for i, height in enumerate(heights):
        tree = Tree()
        # half grow and half full
        if i % 2 == 0:
            # grow method
            tree.grow(max_depth=height)
        else:
            # full method
            tree.full(max_depth=height)
        population.append(tree)
    return population
