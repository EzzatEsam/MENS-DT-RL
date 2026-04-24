from gymnasium import Env
from numpy.typing import ArrayLike


from typing import Literal

MutationType = Literal[
    "Replace_with_child",
    "Truncate",
    "Insert_inner_node",
    "Expand_leaf",
    "Reset_split",
    "Modify_threshold",
    "Modify_leaf",
]


class DecisionTree:
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

    def predict(self, state: ArrayLike) -> int | float | list[float]:
        """
        Predict the action to take given the current state.

        Parameters
        ----------
        state : ArrayLike
            The current observation/state from the environment.

        Returns
        -------
        int
            The action to be taken by the agent.
        """
        pass

    def fit(self, states: list[ArrayLike], actions: list[int | float | list[float]], max_depth: int = None) -> "DecisionTree":
        """
        Train the decision tree from scratch using the provided dataset.

        Parameters
        ----------
        states : list of ArrayLike
            The observations collected from the environment.
        actions : list of actions
            The expert actions corresponding to the states.
        max_depth : int, optional
            The maximum depth to grow the tree.

        Returns
        -------
        DecisionTree
            The fitted tree (self).
        """
        pass

    def clone(self) -> "DecisionTree":
        """
        Create a deep copy of the current Decision Tree.

        Returns
        -------
        DecisionTree
            A new instance of DecisionTree with the same structure and parameters.
        """
        pass

    def mutate(self, mutation_type: MutationType, *args, **kwargs) -> "DecisionTree":
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
        pass

    def get_size(self) -> int:
        """
        Get the size of the tree, defined as the number of nodes.

        Returns
        -------
        int
            The total number of nodes in the tree.
        """
        pass

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
    max_depth: int, state_space: int, action_space: int, *args, **kwargs
) -> DecisionTree:
    """
    Generates a random valid decision tree.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    state_space : int
        The size of the state space.
    action_space : int
        The size of the action space.
    *args : tuple
        Positional arguments for the tree.
    **kwargs : dict
        Keyword arguments for the tree.

    Returns
    -------
    DecisionTree
        A new instance of DecisionTree with the same structure and parameters.
    """
    pass


def initialize_population(
    mode: str, pop_size: int, env: Env, *args, **kwargs
) -> list[DecisionTree]:
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
    list of DecisionTree
        The initialized population of decision trees.
    """
    pass
