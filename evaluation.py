from decision_tree import DecisionTree
from gymnasium import Env


def simulate_episode(tree: DecisionTree, env: Env) -> float:
    """
        Run a single episode in the environment using the provided Decision Tree.

        Parameters
        ----------
        tree : DecisionTree
            The decision tree policy to use for selecting actions.
        env : Env
            The gymnasium environment to run the episode in.

        Returns
    -------
        float
            The total accumulated reward from the episode.
    """
    pass


def evaluate_tree_performance(
    tree: DecisionTree, env: Env, n_episodes: int
) -> list[float]:
    """
        Evaluate the tree's performance by running multiple simulation episodes.

        Parameters
        ----------
        tree : DecisionTree
            The decision tree policy to evaluate.
        env : Env
            The gymnasium environment to run simulations in.
        n_episodes : int
            The number of episodes to simulate.

        Returns
    -------
        list of float
            A list containing the total rewards for each simulated episode.
    """
    pass


def calculate_fitness(rewards: list[float], tree_size: int, alpha: float) -> float:
    """
        Calculate the fitness score for a decision tree.

        The fitness metric rewards trees with high average performance,
        consistent behavior (low standard deviation), and smaller sizes
        (regularization).

        Fitness = Mean(Rewards) - Std(Rewards) - (Alpha * TreeSize)

        Parameters
        ----------
        rewards : list of float
            The list of total rewards obtained from evaluation episodes.
        tree_size : int
            The number of nodes in the decision tree.
        alpha : float
            The regularization parameter that penalizes larger tree sizes.

        Returns
    -------
        float
            The calculated fitness score.
    """
    pass
