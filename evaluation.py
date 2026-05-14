import numpy as np
from gymnasium import Env

from decision_tree_model import DecisionTreeModel
from normalizer import ObservationNormalizer


def simulate_episode(
    tree: DecisionTreeModel, env: Env, render: bool = False
) -> float:
    """
    Run a single episode in the environment using the provided Decision Tree.

    At each timestep, observations whose bounds are known (finite) are
    normalized to [-1, 1] before being passed to the tree — following the
    Attribute Normalization procedure described in §3 of the MENS-DT-RL paper.
    Attributes with infinite bounds are passed through unchanged.

    Parameters
    ----------
    tree : DecisionTreeModel
        The decision tree policy to use for selecting actions.
    env : Env
        The gymnasium environment to run the episode in.
    render : bool, optional
        Whether to render the environment during simulation. Default is False.

    Returns
    -------
    float
        The total accumulated undiscounted reward G(M, i) from the episode.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False

    normalizer = ObservationNormalizer(env.observation_space)

    while not (terminated or truncated):
        if render:
            env.render()

        obs_norm = normalizer.normalize(obs)
        if isinstance(obs_norm, (int, float, np.integer, np.floating)):
            obs_norm = np.array([obs_norm])
            
        action = tree.predict(obs_norm)

        # Ask the environment what type of action it expects based on its shape
        if getattr(env.action_space, 'shape', None) and len(env.action_space.shape) > 0:
            # Environment expects an array (e.g., Box space like Pendulum)
            if isinstance(action, (int, float, np.integer, np.floating)):
                formatted_action = np.array([action], dtype=np.float32)
            else:
                formatted_action = np.array(action, dtype=np.float32)
        else:
            # Environment expects a single integer (e.g., Discrete space like FrozenLake)
            if isinstance(action, np.ndarray):
                formatted_action = int(action.item())
            else:
                formatted_action = int(action)

        # Pass the correctly formatted action to the environment
        obs, reward, terminated, truncated, _ = env.step(formatted_action)
    
        
        total_reward += reward

    return total_reward


def evaluate_tree_performance(
    tree: DecisionTreeModel, env: Env, n_episodes: int
) -> list[float]:
    """
    Evaluate the tree's performance by running multiple simulation episodes.

    Parameters
    ----------
    tree : DecisionTreeModel
        The decision tree policy to evaluate.
    env : Env
        The gymnasium environment to run simulations in.
    n_episodes : int
        The number of episodes to simulate.

    Returns
    -------
    list of float
        A list containing the total undiscounted reward G(M, i)
        for each of the n_episodes simulated episodes.
    """
    rewards = [simulate_episode(tree, env) for _ in range(n_episodes)]
    return rewards


def calculate_fitness(rewards: list[float], tree_size: int, alpha: float) -> float:
    """
    Calculate the fitness score for a decision tree — Equation (1) of the paper.

    fitness(M) = mean(G) - std(G) - alpha * |M|

    where G is the list of undiscounted episode rewards, |M| is the number of
    nodes in the tree, and alpha is the regularization coefficient that
    penalizes larger (less interpretable) trees.

    This formulation simultaneously rewards:
      - High average performance  (mean reward)
      - Consistent behaviour      (low standard deviation)
      - Compact representations   (small tree size)

    Standard deviation uses ddof=0 (population std) to match the paper formula:
        std = sqrt( (1/N) * sum( (G_j - mean)^2 ) )

    Parameters
    ----------
    rewards : list of float
        The list of total undiscounted rewards G(M, i) from evaluation episodes.
    tree_size : int
        The number of nodes in the decision tree (||M||).
    alpha : float
        The regularization parameter that penalizes larger tree sizes.
        Typical values per environment (Table 1):
          CartPole      → 0.1
          MountainCar   → 0.1
          LunarLander   → 2.0
          Maize         → 0.5

    Returns
    -------
    float
        The calculated fitness score.
    """
    g = np.array(rewards, dtype=float)
    return float(np.mean(g) - np.std(g, ddof=0) - alpha * tree_size)


def calculate_success_rate(rewards: list[float], threshold: float) -> float:
    """
    Calculate the fraction of episodes that are considered successful — SR(·).

    An episode is successful when its total undiscounted reward is **strictly
    greater than** the environment's predetermined success threshold.
    This matches the paper's exact wording in §4.1:
      - CartPole-v1  : "reward is **larger than** 495"  (line 512)
      - LunarLander  : "reward is **larger than** 200"  (line 530)

    This helper is used by Reward Pruning (Algorithm 1) to provide a more
    robust comparison between trees, guarding against stochastic noise in the
    fitness signal:
        if fitness(M') >= fitness(M)  or  SR(M') >= SR(M): M = M'

    Success thresholds per environment (§4.1):
      - CartPole-v1       : > 495   (paper §4.1.1, line 512)
      - MountainCar-v0    : > -110  (paper §4.1.2, line 521 — NOTE: paper text
                                     says "lesser than -110" which is a typo;
                                     contextually means r > -110, i.e. reaching
                                     the flag in fewer than 110 steps)
      - LunarLander-v2    : > 200   (paper §4.1.3, line 530)
      - MaizeFertilization: >= 60   (paper §4.1.4, line 542 — "a cumulative
                                     reward of 60", interpreted as >= 60)

    Parameters
    ----------
    rewards : list of float
        The list of total episode rewards.
    threshold : float
        The reward threshold. An episode is a success if its total reward
        strictly exceeds this value (r > threshold), except for Maize
        where >= is used due to the paper's ambiguous wording.

    Returns
    -------
    float
        The success rate in [0.0, 1.0] — fraction of successful episodes.
    """
    if not rewards:
        return 0.0
    successes = sum(1 for r in rewards if r > threshold)
    return successes / len(rewards)
