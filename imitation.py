from gymnasium import Env
import torch as T
from decision_tree import DecisionTree


def run_imitation(
    expert_model: T.nn.Module, d_tree: DecisionTree, env: Env, n_episodes: int
) -> DecisionTree:
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
    DecisionTree
        The trained decision tree that imitates the expert.
    """
    pass

