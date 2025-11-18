class TreeNode:
    def __init__(self, value, current_depth):
        self.value = value
        self.left = None
        self.right = None
        self.current_depth = current_depth

    def is_leaf(self) -> bool:
        return not self.left and not self.right
