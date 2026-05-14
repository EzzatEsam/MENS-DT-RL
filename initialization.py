from gymnasium import Env
import gymnasium as gym
from decision_tree_model import DecisionTreeModel, generate_random_tree
from expert_model import get_expert_model
from imitation import initialize_population_imitation
from pruning import reward_pruning


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
    match mode:
        case "R":
            is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
            if len(env.observation_space.shape) > 0:
                input_dim = env.observation_space.shape[0]
            else:
                # If shape is (), it's a Discrete space, which means it provides 1 integer
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
        case "IL":
            expert_model = kwargs.get("expert_model", None)
            if not expert_model:
                expert_model = get_expert_model(env, use_sb3=True)
            return initialize_population_imitation(
                pop_size,
                env,
                expert_model,
                n_episodes=kwargs.get("n_episodes", 10),
                max_depth=max_depth,
                *args,
                **kwargs,
            )

        case "P":
            # For pruning-based initialization, we would first train a large tree and then prune it to
            expert_model = kwargs.get("expert_model", None)
            if not expert_model:
                expert_model = get_expert_model(env, use_sb3=True)
            pop = initialize_population_imitation(
                pop_size,
                env,
                expert_model,
                n_episodes=kwargs.get("n_episodes", 10),
                max_depth=max_depth,
                *args,
                **kwargs,
            )

            return [
                reward_pruning(
                    tree, env, n_episodes=kwargs.get("n_episodes", 10), *args, **kwargs
                )
                for tree in pop
            ]
