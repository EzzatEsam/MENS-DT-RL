from abc import ABC, abstractmethod
from copy import deepcopy


class TreeNode(ABC):

    @abstractmethod
    def predict(self, state):
        pass

    def clone(self):
        return deepcopy(self)

    @abstractmethod
    def get_n_children(self) -> int:
        pass


class LeafNode(TreeNode):

    def __init__(self, action):
        self.action = action

    def predict(self, state):
        return self.action

    def get_n_children(self):
        return 0


class DecisionNode(TreeNode):

    def __init__(
        self,
        attribute_index: int,
        threshold: float,
        left_child: TreeNode,
        right_child: TreeNode,
    ):
        self.attribute_index = attribute_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child

    def predict(self, state):
        if state[self.attribute_index] <= self.threshold:
            return self.left_child.predict(state)
        else:
            return self.right_child.predict(state)

    def get_n_children(self):
        return 2 + self.left_child.get_n_children() + self.right_child.get_n_children()


def print_tree(node: TreeNode, depth: int = 0, feature_names: list = None):
    """
    Prints the structure of the decision tree.
    """
    indent = "  " * depth
    if isinstance(node, LeafNode):
        print(f"{indent}Leaf: action={node.action}")
    elif isinstance(node, DecisionNode):
        if feature_names and node.attribute_index < len(feature_names):
            feature = feature_names[node.attribute_index]
        else:
            feature = f"state[{node.attribute_index}]"
        print(f"{indent}Decision: {feature} <= {node.threshold}")
        print_tree(node.left_child, depth + 1, feature_names)
        print_tree(node.right_child, depth + 1, feature_names)
