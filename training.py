from decision_tree_model import DecisionTreeModel, initialize_population
from evaluation import evaluate_tree_performance, calculate_fitness
from gymnasium import Env


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
    population = initialize_population(
        mode=init_mode, pop_size=pop_size, env=env, *args, **kwargs
    )

    best_scores = []
    avg_scores = []

    # 2. Evolutionary Loop
    for generation in range(max_generations):
        new_population = []

        # 3. Application of Operators (Multi-method Ensemble)
        for tree in population:
            # We clone the parent and apply a random mutation to create an offspring
            offspring = tree.clone().mutate()
            new_population.append(offspring)

        # 4. Fitness Calculation
        # Combine old and new individuals for evaluation
        combined_candidates = population + new_population
        for tree in combined_candidates:
            # Optimization: Only evaluate if fitness has not been computed yet
            try:
                if tree.get_fitness() is not None:
                    continue
            except (AttributeError, KeyError):
                # Fitness not set yet
                pass

            rewards = evaluate_tree_performance(tree, env, n_episodes)
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
    pass
