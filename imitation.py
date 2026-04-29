from gymnasium import Env
import torch as T
from decision_tree_model import DecisionTreeModel


def run_imitation(
    expert_model: T.nn.Module, env: Env, n_episodes: int
) -> DecisionTreeModel:
    """
    Train a decision tree to imitate an expert model using Imitation Learning.

    Typically uses the DAgger algorithm to collect a dataset of expert actions
    on states visited by the student (tree) and then fits the tree to this data.

    Parameters
    ----------
    expert_model : torch.nn.Module
        The pre-trained expert model (e.g., a Neural Network) to imitate.
    d_tree : DecisionTree
        The decision tree instance to be trained.
    env : Env
        The gymnasium environment used for data collection.
    n_episodes : int
        The number of DAgger iterations or episodes to run.

    Returns
    -------
    DecisionTreeModel
        The trained decision tree that imitates the expert.
    """
    pass


def initialize_population_imitation(
    pop_size: int, env: Env, expert_model: T.nn.Module, n_episodes: int
) -> list[DecisionTreeModel]:
    """
    Initialize the population using Imitation Learning.

    This function creates an initial population of decision trees by training each
    tree to imitate the expert model using the run_imitation function.

    Parameters
    ----------
    pop_size : int
        The number of decision trees to initialize in the population.
    env : Env
        The gymnasium environment used for data collection during imitation learning.
    expert_model : torch.nn.Module
        The pre-trained expert model to imitate.
    n_episodes : int
        The number of episodes to run for each tree's imitation learning.

    Returns
    -------
    list[DecisionTreeModel]
        A list of initialized decision trees trained via imitation learning.
    """
    population = []
    for _ in range(pop_size):
        tree = run_imitation(expert_model, env, n_episodes)
        population.append(tree)
    return population
