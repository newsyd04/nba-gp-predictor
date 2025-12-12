class TreeNode:
    # Initializes a tree node with a value and its current depth
    def __init__(self, value, current_depth):
        self.value = value
        self.left = None
        self.right = None
        self.current_depth = current_depth

    # Checks if the node is a leaf node
    def is_leaf(self) -> bool:
        return not self.left and not self.right
