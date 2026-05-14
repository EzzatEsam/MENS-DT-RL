from decision_tree_model import DecisionTreeModel
from evaluation import evaluate_tree_performance, calculate_fitness
from gymnasium import Env
import random
import math

from initialization import initialize_random_population

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

    # Extract Island Model parameters from kwargs
    num_islands = kwargs.get("num_islands", 4)
    migration_interval = kwargs.get("migration_interval", 5)

    # 1. Initialization Phase
    # Calculate sizes for each island
    island_sizes = [pop_size // num_islands] * num_islands
    # Distribute the remainder
    for i in range(pop_size % num_islands):
        island_sizes[i] += 1

    # Initialize the entire population once
    full_population = initialize_random_population(
        init_mode,
        pop_size,
        env,
        15,
    )

    # Distribute into islands
    islands = []
    start_idx = 0
    for size in island_sizes:
        island_pop = full_population[start_idx : start_idx + size]
        islands.append(island_pop)
        start_idx += size

    best_scores = []
    avg_scores = []

    # Optional callback to save progress each generation
    save_callback = kwargs.get("save_callback", None)

    # 2. Evolutionary Loop
    for generation in range(max_generations):
        mutated_alpha = alpha * (max_generations - generation) / max_generations

        # Evolve each island separately
        for island_idx in range(num_islands):
            population = islands[island_idx]
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
                fitness_score = calculate_fitness(rewards, tree_size, mutated_alpha)
                tree.set_fitness(fitness_score)

            # 5. Selection
            # Sort by fitness (highest first) and keep top individuals
            combined_candidates.sort(key=lambda x: x.get_fitness(), reverse=True)
            islands[island_idx] = combined_candidates[: island_sizes[island_idx]]

        # Migration Event
        if (generation + 1) % migration_interval == 0 and num_islands > 1:
            # Extract the best tree from each island
            best_trees_to_migrate = []
            for island_idx in range(num_islands):
                # Pop the best tree (index 0 since the list is sorted by fitness)
                best_tree = islands[island_idx].pop(0)
                best_trees_to_migrate.append(best_tree)

            # Send the best tree from each island to a random different island
            sources = list(range(num_islands))
            targets = list(range(num_islands))
            # Create a derangement so no tree goes back to its original island
            while any(s == t for s, t in zip(sources, targets)):
                random.shuffle(targets)

            for source_idx in range(num_islands):
                target_island = targets[source_idx]
                incoming_tree = best_trees_to_migrate[source_idx]
                islands[target_island].append(incoming_tree)
                # Re-sort the island to place the incoming tree in the correct rank
                islands[target_island].sort(key=lambda x: x.get_fitness(), reverse=True)

        # Log progress and history
        # Best fitness across all islands
        current_best = max(island[0].get_fitness() for island in islands)
        # Average fitness across all trees in all islands
        total_fitness = sum(sum(t.get_fitness() for t in island) for island in islands)
        avg_fitness = total_fitness / pop_size

        best_scores.append(current_best)
        avg_scores.append(avg_fitness)

        print(
            f"Generation {generation + 1}/{max_generations} - Best Fitness: {current_best:.4f} | Avg Fitness: {avg_fitness:.4f}"
        )

        if save_callback is not None:
            global_best_tree = max(
                (island[0] for island in islands), key=lambda x: x.get_fitness()
            )
            save_callback(global_best_tree, best_scores, avg_scores)

    # 6. Return best solution and history
    global_best_tree = max(
        (island[0] for island in islands), key=lambda x: x.get_fitness()
    )
    return global_best_tree, best_scores, avg_scores
