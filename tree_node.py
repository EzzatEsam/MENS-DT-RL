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


def print_tree(
    node: TreeNode,
    depth: int = 0,
    feature_names: list = None,
    action_names: list = None,
    decimals: int = 3,
):
    """
    Prints the structure of the decision tree.
    """
    indent = "  " * depth
    if isinstance(node, LeafNode):
        action_label = node.action
        if (
            action_names
            and isinstance(node.action, int)
            and node.action < len(action_names)
        ):
            action_label = action_names[node.action]
        print(f"{indent}Leaf: action={action_label}")
    elif isinstance(node, DecisionNode):
        if feature_names and node.attribute_index < len(feature_names):
            feature = feature_names[node.attribute_index]
        else:
            feature = f"state[{node.attribute_index}]"
        threshold = round(float(node.threshold), decimals)
        print(f"{indent}Decision: {feature} <= {threshold}")
        print_tree(node.left_child, depth + 1, feature_names, action_names, decimals)
        print_tree(node.right_child, depth + 1, feature_names, action_names, decimals)


def tree_to_mermaid(
    node: TreeNode,
    feature_names: list = None,
    action_names: list = None,
    decimals: int = 3,
) -> str:
    """
    Generate Mermaid graph text for the decision tree.
    """
    lines = ["graph TD"]
    counter = {"value": 0}

    def next_id() -> str:
        counter["value"] += 1
        return f"n{counter['value']}"

    def format_feature(decision_node: DecisionNode) -> str:
        if feature_names and decision_node.attribute_index < len(feature_names):
            return feature_names[decision_node.attribute_index]
        return f"state[{decision_node.attribute_index}]"

    def label_text(text: str) -> str:
        return text.replace('"', '\\"')

    def walk(current: TreeNode) -> str:
        node_id = next_id()
        if isinstance(current, LeafNode):
            action_label = current.action
            if (
                action_names
                and isinstance(current.action, int)
                and current.action < len(action_names)
            ):
                action_label = action_names[current.action]
            label = label_text(f"Leaf: action={action_label}")
            lines.append(f'{node_id}["{label}"]')
            return node_id

        if isinstance(current, DecisionNode):
            feature = format_feature(current)
            threshold = round(float(current.threshold), decimals)
            label = label_text(f"Decision: {feature} <= {threshold}")
            lines.append(f'{node_id}{{"{label}"}}')

            left_id = walk(current.left_child)
            right_id = walk(current.right_child)

            lines.append(f"{node_id} -- left --> {left_id}")
            lines.append(f"{node_id} -- right --> {right_id}")
            return node_id

        label = label_text("Unknown")
        lines.append(f'{node_id}["{label}"]')
        return node_id

    walk(node)
    return "\n".join(lines)
