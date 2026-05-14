from copy import deepcopy
import random
import numpy as np
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Literal, TypedDict
from tree_node import DecisionNode, LeafNode, TreeNode
from gymnasium import Env
MutationType = Literal[
    "Replace_with_child",
    "Truncate",
    "Insert_inner_node",
    "Expand_leaf",
    "Reset_split",
    "Modify_threshold",
    "Modify_leaf",
]


class DecisionTreeModel:
    """
    An interpretable univariate decision tree policy for reinforcement learning.

    This class implements a decision tree that takes environmental states as input
    and predicts discrete or continuous actions. It is designed to be evolved
    using evolutionary algorithms like MENS-DT-RL.

    Methods
    -------
    predict(state)
        Takes a state and returns the predicted action.
    clone()
        Creates a deep copy of the tree.
    mutate(mutation_type, *args, **kwargs)
        Applies a mutation operator to the tree's structure or parameters.
    fit(states, actions, max_depth)
        Trains the tree from scratch using CART principles (Gini/MSE) on the given data.
    """

    class NodeInfo(TypedDict):
        node: DecisionNode | LeafNode
        parent: DecisionNode | None
        type: Literal["leaf", "decision"]
        relation: Literal["root", "left", "right"]

    def __init__(
        self,
        root: TreeNode,
        is_discrete: bool,
        input_dim: int,
        output_classes: int = None,
        output_range: tuple[float, float] = None,
    ):
        self.root: TreeNode = root
        self.fitness: float = np.nan
        self.is_discrete: bool = is_discrete
        self.input_dim: int = input_dim
        self.output_classes: int = output_classes
        self.output_range: tuple[float, float] = output_range
        self._create_nodes_map()

    def _create_nodes_map(self):
        nodes: list[DecisionTreeModel.NodeInfo] = []

        def traverse(node, parent=None):
            if parent is not None:
                if node is parent.left_child:
                    relation = "left"
                elif node is parent.right_child:
                    relation = "right"
                else:
                    relation = "unknown"
            else:
                relation = "root"
            if isinstance(node, LeafNode):
                nodes.append(
                    {
                        "node": node,
                        "parent": parent,
                        "type": "leaf",
                        "relation": relation,
                    }
                )
            elif isinstance(node, DecisionNode):
                nodes.append(
                    {
                        "node": node,
                        "parent": parent,
                        "type": "decision",
                        "relation": relation,
                    }
                )
                traverse(node.left_child, parent=node)
                traverse(node.right_child, parent=node)

        traverse(self.root)
        self.nodes_map = nodes

    def _get_random_node(
        self, node_type: str | set[str], filter_root: bool = False
    ) -> NodeInfo | None:
        if isinstance(node_type, str):
            candidates = [n for n in self.nodes_map if n["type"] == node_type]
        else:
            candidates = [n for n in self.nodes_map if n["type"] in node_type]
        if filter_root:
            candidates = [n for n in candidates if n["relation"] != "root"]
        return random.choice(candidates) if candidates else None

    def _get_random_output_value(self) -> int | float:
        if self.is_discrete:
            return np.random.randint(self.output_classes)
        else:
            low, high = self.output_range
            return np.random.uniform(low, high)

    def _swap_node_in_tree(self, node: NodeInfo, new_node: TreeNode) -> None:
        parent = node["parent"]
        relation = node["relation"]
        if relation == "left":
            parent.left_child = new_node
        elif relation == "right":
            parent.right_child = new_node
        elif relation == "root":
            self.root = new_node

    def predict(self, state: ArrayLike) -> int | float:
        """
        Predict the action to take given the current state.

        Parameters
        ----------
        state : ArrayLike
            The current observation/state from the environment.

        Returns
        -------
        int | float
            The action to be taken by the agent.
        """
        return self.root.predict(state)

    @classmethod
    def _build_tree_from_sklearn(cls, sklearn_tree) -> TreeNode:
        def build_node(node_id: int) -> TreeNode:
            # In scikit-learn, leaf nodes have left and right children set to -1
            is_leaf = (
                sklearn_tree.children_left[node_id]
                == sklearn_tree.children_right[node_id]
            )

            if is_leaf:
                # Extract the value array for this node
                value_array = sklearn_tree.value[node_id][0]

                # If the array has more than 1 element, it's a classification tree (discrete)
                # and contains class frequencies. We take the argmax for the predicted class.
                if len(value_array) > 1:
                    action = int(np.argmax(value_array))
                # Otherwise, it's a regression tree (continuous) with a single predicted value.
                else:
                    action = float(value_array[0])

                return LeafNode(action=action)
            else:
                # It is a decision node
                feature_idx = int(sklearn_tree.feature[node_id])
                threshold = float(sklearn_tree.threshold[node_id])

                # Recursively build the subtrees
                left_child = build_node(sklearn_tree.children_left[node_id])
                right_child = build_node(sklearn_tree.children_right[node_id])

                return DecisionNode(
                    attribute_index=feature_idx,
                    threshold=threshold,
                    left_child=left_child,
                    right_child=right_child,
                )

        # Scikit-learn's root node is always initialized at index 0
        return build_node(0)

    @classmethod
    def fit(
        cls,
        states: list[ArrayLike],
        actions: list[int | float],
        is_discrete: bool,
        input_dim: int,
        max_depth: int = None,
        output_classes: int = None,
        output_range: tuple[float, float] = None,
    ) -> "DecisionTreeModel":
        """
        Train the decision tree from scratch using the provided dataset.

        Parameters
        ----------
        states : list of ArrayLike
            The observations collected from the environment.
        actions : list of actions
            The expert actions corresponding to the states.
        is_discrete : bool
            Whether the output is discrete.
        input_dim : int
            The dimensionality of the input features.
        max_depth : int, optional
            The maximum depth to grow the tree.
        output_classes : int, optional
            The number of classes for discrete outputs.
        output_range : tuple of float, optional
            The range of values for continuous outputs.

        Returns
        -------
        DecisionTreeModel
            The fitted tree (self).
        """

        if is_discrete:
            clf = DecisionTreeClassifier(max_depth=max_depth)
        else:
            clf = DecisionTreeRegressor(max_depth=max_depth)

        clf.fit(states, actions)

        # Build our custom tree structure from the trained scikit-learn tree
        root = cls._build_tree_from_sklearn(clf.tree_)

        return cls(
            root=root,
            is_discrete=is_discrete,
            input_dim=input_dim,
            output_classes=output_classes,
            output_range=output_range,
        )

    def clone(self) -> "DecisionTreeModel":
        """
        Create a deep copy of the current Decision Tree.

        Returns
        -------
        DecisionTree
            A new instance of DecisionTree with the same structure and parameters.
        """
        other = deepcopy(self)
        other.set_fitness(np.nan)  # Reset fitness for the cloned tree
        return other

    def mutate(
        self, mutation_type: MutationType, *args, **kwargs
    ) -> "DecisionTreeModel":
        """
        Apply a mutation operator to the tree structure or parameters.

        This method should implement various mutation strategies such as
        pruning, expanding leaves, or modifying thresholds as described
        in the MENS-DT-RL algorithm.

        Parameters
        ----------
        mutation_type : MutationType
            The type of mutation to apply.
        *args : tuple
            Positional arguments for mutation.
        **kwargs : dict
            Keyword arguments for mutation.

        Returns
        -------
        DecisionTree
            The mutated tree (or a new instance if modified in a non-in-place manner).
        """
        match mutation_type:
            case "Replace_with_child":
                # Select a random node and replace it with one of its children.
                # If the selected node is the root, replace the entire tree with one of its children.
                random_des_node = self._get_random_node("decision", filter_root=False)
                p = np.random.rand()
                if p < 0.5:
                    new_child = random_des_node["node"].left_child
                else:
                    new_child = random_des_node["node"].right_child

                self._swap_node_in_tree(random_des_node, new_child)
            case "Reset_split":
                # Select a random decision node and reset its split parameters to new random values.
                random_des_node = self._get_random_node("decision", filter_root=False)
                random_des_node["node"].attribute_index = np.random.randint(
                    self.input_dim
                )
                random_des_node["node"].threshold = np.random.uniform(-1, 1)

            case "Truncate":
                # Select a random decision node and replace it with a leaf node.
                random_des_node = self._get_random_node("decision", filter_root=False)
                new_leaf = LeafNode(action=self._get_random_output_value())
                self._swap_node_in_tree(random_des_node, new_leaf)
            case "Modify_threshold":
                # Select a random decision node and modify its threshold by adding a small random perturbation.
                random_des_node = self._get_random_node("decision", filter_root=False)
                perturbation = np.random.normal(loc=0.0, scale=0.1)
                random_des_node["node"].threshold += perturbation
                # Ensure the threshold remains within the normalized range [-1, 1].
                random_des_node["node"].threshold = np.clip(
                    random_des_node["node"].threshold, -1, 1
                )
            case "Insert_inner_node":
                # Insert a new random decision node above a randomly selected node (which becomes one of its children).
                # The other child of the new decision node is a new random leaf node.
                random_node = self._get_random_node(
                    {"decision", "leaf"}, filter_root=False
                )
                rand_attribute_index = np.random.randint(self.input_dim)
                rand_threshold = np.random.uniform(-1, 1)

                p = np.random.rand()
                if p < 0.5:
                    new_decision_node = DecisionNode(
                        attribute_index=rand_attribute_index,
                        threshold=rand_threshold,
                        left_child=random_node["node"],
                        right_child=LeafNode(action=self._get_random_output_value()),
                    )
                else:
                    new_decision_node = DecisionNode(
                        attribute_index=rand_attribute_index,
                        threshold=rand_threshold,
                        left_child=LeafNode(action=self._get_random_output_value()),
                        right_child=random_node["node"],
                    )

                self._swap_node_in_tree(random_node, new_decision_node)
            case "Expand_leaf":
                # Select a random leaf node and replace it with a new decision node that splits on a random attribute and threshold.
                # The two children of the new decision node are new random leaf nodes.
                random_leaf_node = self._get_random_node("leaf", filter_root=False)
                rand_attribute_index = np.random.randint(self.input_dim)
                rand_threshold = np.random.uniform(-1, 1)

                new_decision_node = DecisionNode(
                    attribute_index=rand_attribute_index,
                    threshold=rand_threshold,
                    left_child=LeafNode(action=self._get_random_output_value()),
                    right_child=LeafNode(action=self._get_random_output_value()),
                )

                self._swap_node_in_tree(random_leaf_node, new_decision_node)
            case "Modify_leaf":
                # Select a random leaf node and modify its action to a new random value.
                random_leaf_node = self._get_random_node("leaf", filter_root=False)
                random_leaf_node["node"].action = self._get_random_output_value()
            case _:
                raise ValueError(f"Unsupported mutation type: {mutation_type}")

        self._create_nodes_map()  # Rebuild the nodes map after mutation to reflect changes.
        return self

    def get_size(self) -> int:
        """
        Get the size of the tree, defined as the number of nodes.

        Returns
        -------
        int
            The total number of nodes in the tree.
        """
        return 1 + self.root.get_n_children() if self.root else 0

    def set_fitness(self, fitness: float) -> None:
        """
        Set the fitness of the tree.

        Parameters
        ----------
        fitness : float
            The fitness of the tree.
        """
        self.fitness = fitness

    def get_fitness(self) -> float:
        """
        Get the fitness of the tree.

        Returns
        -------
        float
            The fitness of the tree.
        """
        return self.fitness

    ### Other needed methods will be added


def generate_random_tree(
    max_depth: int,
    state_space: int,
    is_discrete: bool,
    output_classes: int = None,
    output_range: tuple[float, float] = None,
    early_stop_prob: float = 0.2,
    *args,
    **kwargs,
) -> DecisionTreeModel:
    """
    Generates a random valid decision tree.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    state_space : int
        The size of the state space.
    is_discrete : bool
        Whether the output is discrete.
    output_classes : int, optional
        The number of classes for discrete outputs.
    output_range : tuple of float, optional
        The range of values for continuous outputs.
    early_stop_prob : float, optional
        The probability of early stopping to create variability in tree sizes.
    *args : tuple
        Positional arguments for the tree.
    **kwargs : dict
        Keyword arguments for the tree.

    Returns
    -------
    DecisionTreeModel
        A new instance of DecisionTreeModel with the same structure and parameters.
    """

    def _get_random_action() -> int | float:
        """Helper to generate a random action based on the environment type."""
        if is_discrete:
            return np.random.randint(output_classes)
        else:
            low, high = output_range
            return np.random.uniform(low, high)

    def _build_tree(current_depth: int):
        """Recursively builds the tree structure."""
        # 1. Base Case: Reached maximum depth, must cap with a leaf.
        if current_depth >= max_depth:
            return LeafNode(action=_get_random_action())

        # 2. Early Stopping: Give it a chance (e.g., 20%) to randomly stop growing
        # and become a leaf, ensuring trees aren't all uniform in size.
        # We ensure depth > 0 so the root isn't immediately a leaf.
        if current_depth > 0 and np.random.rand() < early_stop_prob:
            return LeafNode(action=_get_random_action())

        # 3. Create Inner Rule (Decision Node)
        # Select a random feature index
        rand_attribute_index = np.random.randint(state_space)
        # Select a random threshold in [-1, 1] (assuming inputs are normalized)
        rand_threshold = np.random.uniform(-1.0, 1.0)

        # Recursively build left and right subtrees
        left = _build_tree(current_depth + 1)
        right = _build_tree(current_depth + 1)

        return DecisionNode(
            attribute_index=rand_attribute_index,
            threshold=rand_threshold,
            left_child=left,
            right_child=right,
        )

    # Kick off the recursive generation starting at depth 0
    root = _build_tree(0)

    return DecisionTreeModel(
        root=root,
        is_discrete=is_discrete,
        input_dim=state_space,
        output_classes=output_classes,
        output_range=output_range,
    )


def initialize_random_population(
    mode: str, pop_size: int, env: Env, max_depth: int, *args, **kwargs
) -> list[DecisionTreeModel]:
    """
    Orchestrate the chosen initialization mode for the population.

    Supports Random (R), Imitation Learning (IL), and Pruning-based (P)
    initialization strategies as described in the MENS-DT-RL paper.

    Parameters
    ----------
    mode : str
        The initialization mode: 'R', 'IL', or 'P'.
    pop_size : int
        The number of individuals to generate for the initial population.
    env : Env
        The gymnasium environment to use for initialization context.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments (e.g., expert_path).

    Returns
    -------
    list of DecisionTreeModel
        The initialized population of decision trees.
    """
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs_shape = getattr(env.observation_space, 'shape', None)
    if obs_shape is not None and len(obs_shape) > 0:
        input_dim = obs_shape[0]
    else:
        # Fallback for Discrete spaces like FrozenLake or Blackjack
        input_dim = 1
    output_classes = env.action_space.n if is_discrete else None
    # output_range = env.action_space.low[0], (
    #     env.action_space.high[0] if not is_discrete else None
    # )

    if is_discrete:
        output_range = None
    else:
        output_range = (env.action_space.low[0], env.action_space.high[0])
    return [
        generate_random_tree(
            max_depth=max_depth,
            state_space=input_dim,
            is_discrete=is_discrete,
            output_classes=output_classes,
            output_range=output_range,
            *args,
            **kwargs,
        )
        for _ in range(pop_size)
    ]
