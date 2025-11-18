import random
from tree.tree_node import TreeNode
from gp_config import OPERATIONS, TERMINALS

class Tree:
    def __init__(self, node_value=None, current_depth=1):
        self.root = TreeNode(node_value, current_depth=current_depth)

    def to_string(self) -> str:
        return self._to_string_helper(self.root)

    def _to_string_helper(self, node: TreeNode) -> str:
        if node is None:
            return ""
        if node.is_leaf():
            return str(node.value)
        left_str = self._to_string_helper(node.left)
        right_str = self._to_string_helper(node.right)
        return f"({left_str}{node.value}{right_str})"

    def print_tree(self):
        print(self.to_string())

    def random_node(self) -> TreeNode:
        nodes = self._collect_nodes(self.root)
        return random.choice(nodes) if nodes else self.root

    def _collect_nodes(self, node: TreeNode) -> list[TreeNode]:
        if node is None:
            return []
        nodes = [node]
        nodes += self._collect_nodes(node.left)
        nodes += self._collect_nodes(node.right)
        return nodes

    def grow(self, max_depth: int, current_depth=1):
        self.root = TreeNode(None, current_depth=current_depth)
        self._grow(self.root, current_depth, max_depth)

    def _grow(self, node: TreeNode, current_depth: int, max_depth: int, operations: list[str]):
        if current_depth < max_depth and random.random() < 0.7:
                node.value = random.choice(OPERATIONS)
                node.left = TreeNode(None, current_depth + 1)
                node.right = TreeNode(None, current_depth + 1)
                self._grow(node.left, current_depth + 1, max_depth)
                self._grow(node.right, current_depth + 1, max_depth)
        else:
            node.value = random.choice(TERMINALS)

    def full(self, max_depth: int):
        self.root = TreeNode(None, current_depth=1)
        self._full(self.root, 1, max_depth)

    def _full(self, node: TreeNode, current_depth: int, max_depth: int):
        if current_depth < max_depth:
            node.value = random.choice(OPERATIONS)
            node.left = TreeNode(None, current_depth + 1)
            node.right = TreeNode(None, current_depth + 1)
            self._full(node.left, current_depth + 1, max_depth)
            self._full(node.right, current_depth + 1, max_depth)
        else:
            node.value = random.choice(TERMINALS)
