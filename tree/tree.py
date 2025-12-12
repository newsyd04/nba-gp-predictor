import random
from tree.tree_node import TreeNode
from config import OPERATIONS, TERMINALS, ROLLING_COLUMNS

# Define unary and binary operations
UNARY_OPS = ["abs(x)", "log(|x| + 1)", "tanh", "relu"]
BINARY_OPS = ['+', '-', '*', '/']

class Tree:
    def __init__(self, node_value=None, current_depth=1, variables=None):
        self.root = TreeNode(node_value, current_depth=current_depth)
        self.variables = variables

    def to_string(self) -> str:
        return self._to_string_helper(self.root)

    def _to_string_helper(self, node: TreeNode) -> str:
        if node is None:
            return ""
        if node.is_leaf():
            return str(node.value)
        
        if node.value in UNARY_OPS:
            return f"{node.value.replace('(x)', '')}({self._to_string_helper(node.left)})"

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
    
    def find_parent(self, target: TreeNode) -> TreeNode | None:
        if self.root is target:
            return None
        return self._find_parent_helper(self.root, target)

    def _find_parent_helper(self, current: TreeNode, target: TreeNode) -> TreeNode | None:
        if current is None:
            return None
        if current.left is target or current.right is target:
            return current

        left = self._find_parent_helper(current.left, target)
        if left:
            return left

        return self._find_parent_helper(current.right, target)

    def grow(self, max_depth: int, current_depth=1):
        self.root = TreeNode(None, current_depth=current_depth)
        self._grow(self.root, current_depth, max_depth)

    def _grow(self, node: TreeNode, current_depth: int, max_depth: int):
        if current_depth < max_depth and random.random() < 0.7:
            op = random.choice(OPERATIONS)
            node.value = op

            if op in UNARY_OPS:
                node.left = TreeNode(None, current_depth + 1)
                node.right = None
                self._grow(node.left, current_depth + 1, max_depth)
            else:  # binary
                node.left = TreeNode(None, current_depth + 1)
                node.right = TreeNode(None, current_depth + 1)
                self._grow(node.left, current_depth + 1, max_depth)
                self._grow(node.right, current_depth + 1, max_depth)
            return

        choice = random.choice(TERMINALS)
        if choice == 'var':
            node.value = random.choice(self.variables)
        else:
            node.value = round(random.uniform(-10, 10), 2)

    def full(self, max_depth: int):
        self.root = TreeNode(None, current_depth=1)
        self._full(self.root, 1, max_depth)

    def _full(self, node: TreeNode, current_depth: int, max_depth: int):
        if current_depth < max_depth:
            op = random.choice(OPERATIONS)
            node.value = op

            if op in UNARY_OPS:
                node.left = TreeNode(None, current_depth + 1)
                node.right = None
                self._full(node.left, current_depth + 1, max_depth)
            else:
                node.left = TreeNode(None, current_depth + 1)
                node.right = TreeNode(None, current_depth + 1)
                self._full(node.left, current_depth + 1, max_depth)
                self._full(node.right, current_depth + 1, max_depth)
        else:
            choice = random.choice(TERMINALS)
            if choice == 'var':
                node.value = random.choice(self.variables)
            else:
                node.value = round(random.uniform(-10, 10), 2)
                
    def size(self) -> int:
        return len(self._collect_nodes(self.root))

    def depth(self) -> int:
        return self._compute_depth(self.root)

    def _compute_depth(self, node: TreeNode):
        if node is None:
            return 0
        if node.is_leaf():
            return node.current_depth
        return max(self._compute_depth(node.left),
                   self._compute_depth(node.right))