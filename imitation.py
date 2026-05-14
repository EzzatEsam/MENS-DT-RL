from gymnasium import Env
import gymnasium as gym
import torch as T
from decision_tree_model import DecisionTreeModel
from evaluation import evaluate_tree_performance
import numpy as np

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
    
    states = []
    actions = []
    
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    input_dim = env.observation_space.shape[0]
    output_classes = env.action_space.n if is_discrete else None
    output_range = (env.action_space.low[0], env.action_space.high[0]) if not is_discrete else None
    
    current_tree = None
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        
        # 1. Roll out the current student tree (or expert on the first episode) to collect visited states
        while not (terminated or truncated):
            states.append(obs)
            
            # 2. Query the expert model to label those states with expert actions
            expert_action, _ = expert_model.predict(obs)
            if is_discrete:
                actions.append(int(expert_action))
            else:
                actions.append(float(expert_action))
            
            # Agent steps in environment
            if current_tree is None:
                action_to_take = expert_action
            else:
                action_to_take = current_tree.predict(obs)
                
            obs, reward, terminated, truncated, _ = env.step(action_to_take)
            
        # 3 & 4. Aggregate dataset and fit tree with an elbow-method regularization sweep
        best_tree = None
        best_perf = -float('inf')
        
        # Sweep max_depth to find the smallest tree that maintains performance
        for depth in range(1, 5):
            candidate_tree = DecisionTreeModel.fit(
                states=states,
                actions=actions,
                is_discrete=is_discrete,
                input_dim=input_dim,
                max_depth=depth,
                output_classes=output_classes,
                output_range=output_range
            )
            
            # Evaluate performance
            rewards = evaluate_tree_performance(candidate_tree, env, n_episodes=3)
            perf = np.mean(rewards)
            
            # Accept if performance improves; break if it negatively degrades (elbow hit)
            if best_tree is None or perf >= best_perf:
                best_tree = candidate_tree
                best_perf = perf
            elif perf < best_perf - 5.0: 
                break 
                
        current_tree = best_tree
        
    return current_tree


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
    expert_model  = expert_model if expert_model else DummyExpert(env.action_space)
    population = []
    for _ in range(pop_size):
        tree = run_imitation(expert_model, env, n_episodes)
        population.append(tree)
    return population


import torch.nn as nn
class DummyExpert(nn.Module):
    """
    A fake expert model to test the DAgger loop without 
    needing a real trained SB3 model.
    """
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def predict(self, obs):
        # Simply return a random action valid for the environment
        action = self.action_space.sample()
        return action, None
