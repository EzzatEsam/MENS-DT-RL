
from gymnasium import Env

from decision_tree_model import DecisionTreeModel
from evaluation import calculate_fitness, calculate_success_rate, evaluate_tree_performance
from tree_node import LeafNode


def reward_pruning(
    tree: DecisionTreeModel, env: Env, old_score: float = None, *args, **kwargs
) -> DecisionTreeModel:
    """
    Implements the Reward Pruning algorithm (Algorithm 1 from the paper).

    This process iterates over the tree, attempting to prune branches. It checks
    the environment for evaluation (simulating episodes) to verify if the
    pruning operation maintains or improves the old_score (fitness or success rate).

    Parameters
    ----------
    tree : DecisionTree
        The tree to be pruned.
    env : Env
        The environment to use for evaluation during pruning.
    old_score : float, optional
        The previous fitness or success rate score to compare against.

    Returns
    -------
    DecisionTree
        The pruned version of the input tree.
    """
    # Logic to iteratively replace inner nodes with subtrees and check performance against old_score
    n_rounds = kwargs.get("n_rounds", 3)
    n_episodes = kwargs.get("n_episodes", 5)
    alpha = kwargs.get("alpha", 0.1)

    # Establish dynamic success rate threshold depending on environment
    env_id = (
        env.unwrapped.spec.id
        if hasattr(env.unwrapped, "spec") and env.unwrapped.spec
        else ""
    )
    if "CartPole" in env_id:
        threshold = 495.0
    elif "MountainCar" in env_id:
        threshold = -110.0
    elif "LunarLander" in env_id:
        threshold = 200.0
    elif "Maize" in env_id:
        threshold = 60.0
    else:
        threshold = 0.0

    # Evaluate baseline M performance
    rewards_M = evaluate_tree_performance(tree, env, n_episodes)
    fit_M = (
        calculate_fitness(rewards_M, tree.get_size(), alpha)
        if old_score is None
        else old_score
    )
    sr_M = calculate_success_rate(rewards_M, threshold)

    for _ in range(n_rounds):
        # Post-order traversal helper to ensure bottom-up, left-to-right processing
        def get_inner_nodes(node, parent, relation):
            if isinstance(node, LeafNode):
                return []
            nodes = []
            nodes.extend(get_inner_nodes(node.left_child, node, "left"))
            nodes.extend(get_inner_nodes(node.right_child, node, "right"))
            nodes.append({"node": node, "parent": parent, "relation": relation})
            return nodes

        inner_nodes = get_inner_nodes(tree.root, None, "root")

        for node_info in inner_nodes:
            original_node = node_info["node"]

            # Try replacing with Left Child (M')
            left_child = original_node.left_child
            tree._swap_node_in_tree(node_info, left_child)
            tree._create_nodes_map()

            rewards_prime = evaluate_tree_performance(tree, env, n_episodes)
            fit_prime = calculate_fitness(rewards_prime, tree.get_size(), alpha)
            sr_prime = calculate_success_rate(rewards_prime, threshold)

            if fit_prime >= fit_M or sr_prime > sr_M:
                fit_M, sr_M = fit_prime, sr_prime
                continue

            # Revert left child replacement
            tree._swap_node_in_tree(
                {
                    "node": left_child,
                    "parent": node_info["parent"],
                    "relation": node_info["relation"],
                },
                original_node,
            )

            # Try replacing with Right Child (M'')
            right_child = original_node.right_child
            tree._swap_node_in_tree(node_info, right_child)
            tree._create_nodes_map()

            rewards_prime_prime = evaluate_tree_performance(tree, env, n_episodes)
            fit_prime_prime = calculate_fitness(
                rewards_prime_prime, tree.get_size(), alpha
            )
            sr_prime_prime = calculate_success_rate(rewards_prime_prime, threshold)

            if fit_prime_prime >= fit_M or sr_prime_prime > sr_M:
                fit_M, sr_M = fit_prime_prime, sr_prime_prime
                continue

            # Revert right child replacement
            tree._swap_node_in_tree(
                {
                    "node": right_child,
                    "parent": node_info["parent"],
                    "relation": node_info["relation"],
                },
                original_node,
            )
            tree._create_nodes_map()

    tree.set_fitness(fit_M)
    return tree
