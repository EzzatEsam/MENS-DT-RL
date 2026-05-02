from decision_tree_model import DecisionTreeModel, initialize_random_population
from evaluation import evaluate_tree_performance, calculate_fitness
from gymnasium import Env
from evaluation import calculate_success_rate
from tree_node import LeafNode
import random
import math

mutation_types = [
    "Replace_with_child",
    "Truncate",
    "Insert_inner_node",
    "Expand_leaf",
    "Reset_split",
    "Modify_threshold",
    "Modify_leaf",
]


def train_mens_dt_rl(
    env: Env,
    pop_size: int,
    max_generations: int,
    n_episodes: int,
    alpha: float,
    init_mode: str,
    *args,
    **kwargs,
) -> tuple[DecisionTreeModel, list[float], list[float]]:
    """
    Core implementation of the MENS-DT-RL algorithm.

    This function handles the evolutionary loop, applying mutation operators,
    evaluating individuals, and selecting survivors based on the fitness metric.

    Parameters
    ----------
    env : Env
        The Reinforcement Learning environment.
    pop_size : int
        The size of the population in each generation.
    max_generations : int
        The number of generations to evolve the population.
    n_episodes : int
        Number of episodes to run for evaluating a tree's fitness.
    alpha : float
        Regularization parameter to penalize tree size.
    init_mode : str
        Initialization method: "R" (Random), "IL" (Imitation Learning), or "P" (Pruning).

    Returns
    -------
    tuple of (DecisionTree, list[float], list[float])
        The best decision tree model found, history of best scores, and history of average scores.
    """

    # 1. Initialization Phase
    population = initialize_random_population(
        init_mode,
        pop_size,
        env,
        15,
    )
    best_scores = []
    avg_scores = []

    # 2. Evolutionary Loop
    for generation in range(max_generations):
        new_population = []

        # 3. Application of Operators (Multi-method Ensemble)
        for tree in population:
            selected_mutation = random.choice(mutation_types)

            # --- NEW DEFENSIVE LOGIC ---
            # If the tree is just a single leaf (size 1), it has no decision nodes.
            # Force the mutation to be one of the leaf-compatible operators.
            if tree.get_size() <= 1 and selected_mutation not in [
                "Expand_leaf",
                "Modify_leaf",
            ]:
                selected_mutation = random.choice(["Expand_leaf", "Modify_leaf"])

            try:
                # Attempt to mutate the cloned tree
                offspring = tree.clone().mutate(selected_mutation)
            except (TypeError, AttributeError, KeyError) as e:
                # If a structural mutation inexplicably fails due to an edge-case shape,
                # catch the error and simply pass the un-mutated clone to the next generation.
                print(
                    f"Mutation '{selected_mutation}' failed on tree of size {tree.get_size()}. Error: {e}"
                )
                offspring = tree.clone()
            new_population.append(offspring)

        # 4. Fitness Calculation
        # Combine old and new individuals for evaluation
        combined_candidates = population + new_population
        for tree in combined_candidates:

            current_fit = tree.get_fitness()
            # Safely force recalculation if fitness is None or NaN
            if current_fit is not None and not math.isnan(float(current_fit)):
                continue

            # -------------------------

            rewards = evaluate_tree_performance(tree, env, n_episodes)

            # Safety check: Prevent 'nan' if n_episodes was 0 or simulation failed
            if not rewards:
                raise ValueError(
                    "evaluate_tree_performance returned empty rewards. Check n_episodes!"
                )

            tree_size = tree.get_size()
            fitness_score = calculate_fitness(rewards, tree_size, alpha)
            tree.set_fitness(fitness_score)

        # 5. Selection
        # Sort by fitness (highest first) and keep top individuals
        combined_candidates.sort(key=lambda x: x.get_fitness(), reverse=True)
        population = combined_candidates[:pop_size]

        # Log progress and history
        current_best = population[0].get_fitness()
        avg_fitness = sum(t.get_fitness() for t in population) / pop_size

        best_scores.append(current_best)
        avg_scores.append(avg_fitness)

        print(
            f"Generation {generation + 1}/{max_generations} - Best Fitness: {current_best:.4f} | Avg Fitness: {avg_fitness:.4f}"
        )

    # 6. Return best solution and history
    return population[0], best_scores, avg_scores


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
