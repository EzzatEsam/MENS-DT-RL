import gymnasium as gym
import torch.nn as nn
from evaluation import evaluate_tree_performance, calculate_fitness
from imitation import run_imitation
from training import reward_pruning, train_mens_dt_rl
from decision_tree_model import initialize_random_population

# ---------------------------------------------------------
# 1. Dummy Expert for Testing
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. Test Cases
# ---------------------------------------------------------
def test_imitation_learning():
    print("\n=== Testing Phase 1: DAgger Imitation Learning ===")
    env = gym.make("CartPole-v1")
    dummy_expert = DummyExpert(env.action_space)
    
    try:
        # Run for just 2 episodes to see if it executes without crashing
        tree = run_imitation(dummy_expert, env, n_episodes=2)
        print("SUCCESS: Imitation loop completed.")
        print(f"Resulting tree size: {tree.get_size() if tree else 'None'}")
    except Exception as e:
        print(f"FAILED: Imitation loop threw an error: {e}")

def test_reward_pruning():
    print("\n=== Testing Phase 2: Reward Pruning ===")
    env = gym.make("CartPole-v1")
    
    try:
        # Initialize a random tree to act as our baseline 'M'
        population = initialize_random_population(mode="R", pop_size=1, env=env, max_depth=5)
        tree = population[0]
        initial_size = tree.get_size()
        
        # Run pruning for 1 round
        pruned_tree = reward_pruning(tree, env, old_score=None, n_rounds=1, n_episodes=2, alpha=0.1)
        print("SUCCESS: Reward pruning completed.")
        print(f"Tree size before: {initial_size} -> After: {pruned_tree.get_size()}")
    except Exception as e:
        print(f"FAILED: Reward pruning threw an error: {e}")

def test_evolutionary_loop():
    print("\n=== Testing Phase 3: Evolutionary Training Loop ===")
    env = gym.make("CartPole-v1")
    
    try:
        # Run the main loop with Random init to bypass needing an expert path for now
        # Small population and generation count for speed
        best_tree, best_hist, avg_hist = train_mens_dt_rl(
            env=env,
            pop_size=4,
            max_generations=2,
            n_episodes=2,
            alpha=0.1,
            init_mode="R" 
        )
        print("SUCCESS: Evolutionary loop completed.")
        print(f"Final Best Fitness: {best_tree.get_fitness():.4f}")
        print(f"Best Fitness History: {best_hist}")
    except Exception as e:
        print(f"FAILED: Evolutionary loop threw an error: {e}")




# ---------------------------------------------------------
# 1. PREDICTABLE Expert for Correctness Testing
# ---------------------------------------------------------
class PredictableExpert(nn.Module):
    """
    An expert that uses a simple, deterministic rule so we 
    can actually verify if the Decision Tree successfully imitated it.
    """
    def predict(self, obs):
        # CartPole rule: if pole angle (obs[2]) is > 0, move right (1), else move left (0)
        action = 1 if obs[2] > 0 else 0
        return action, None

# ---------------------------------------------------------
# 2. Correctness Tests
# ---------------------------------------------------------
def test_imitation_correctness():
    print("\n=== Testing Phase 1: Imitation Correctness ===")
    env = gym.make("CartPole-v1")
    expert = PredictableExpert()
    
    # Train the tree
    tree = run_imitation(expert, env, n_episodes=5)
    
    # Test how well the tree matches the expert on 100 random states
    matches = 0
    test_states = [env.observation_space.sample() for _ in range(100)]
    
    for state in test_states:
        expert_action, _ = expert.predict(state)
        tree_action = tree.predict(state)
        if expert_action == tree_action:
            matches += 1
            
    accuracy = matches / 100.0
    print(f"Imitation Accuracy: {accuracy * 100}%")
    
    # The tree should have successfully cloned a simple 1-rule expert
    assert accuracy > 0.80, f"Imitation failed! Accuracy too low: {accuracy}"
    print("SUCCESS: Imitation logic is sound.")

def test_pruning_correctness():
    print("\n=== Testing Phase 2: Pruning Correctness ===")
    env = gym.make("CartPole-v1")
    
    # Initialize a deep random tree (max_depth=7) to ensure it has nodes to prune
    population = initialize_random_population(mode="R", pop_size=1, env=env, max_depth=7)
    tree = population[0]
    
    # Get baseline metrics
    initial_size = tree.get_size()
    rewards = evaluate_tree_performance(tree, env, n_episodes=3)
    initial_fitness = calculate_fitness(rewards, initial_size, alpha=0.1)
    tree.set_fitness(initial_fitness)
    
    # Prune
    pruned_tree = reward_pruning(tree, env, old_score=initial_fitness, n_rounds=2, n_episodes=3, alpha=0.1)
    
    final_size = pruned_tree.get_size()
    final_fitness = pruned_tree.get_fitness()
    
    print(f"Size: {initial_size} -> {final_size}")
    print(f"Fitness: {initial_fitness:.2f} -> {final_fitness:.2f}")
    
    # CRITICAL CHECKS:
    assert final_size <= initial_size, "Pruning somehow increased the tree size!"
    assert final_fitness >= initial_fitness, "Pruning degraded the tree's performance/fitness!"
    print("SUCCESS: Pruning logic maintains/improves fitness and reduces size.")

def test_evolution_correctness():
    print("\n=== Testing Phase 3: Evolution Correctness ===")
    env = gym.make("CartPole-v1")
    
    # Run a slightly longer loop to give it time to evolve
    best_tree, best_hist, avg_hist = train_mens_dt_rl(
        env=env,
        pop_size=10,            # Larger population
        max_generations=5,      # Enough generations to see improvement
        n_episodes=2,
        alpha=0.1,
        init_mode="R" 
    )
    
    initial_best = best_hist[0]
    final_best = best_hist[-1]
    
    print(f"Fitness progression: {best_hist}")
    
    # CRITICAL CHECK:
    # Because parents and offspring are combined and the top N are kept, 
    # the best score mathematically CANNOT decrease over time.
    for i in range(1, len(best_hist)):
        assert best_hist[i] >= best_hist[i-1], f"Fitness degraded at generation {i}! Selection logic is flawed."
        
    print("SUCCESS: Evolutionary loop monotonically improves (or maintains) the best fitness.")

    
if __name__ == "__main__":
    print("Starting Unit Tests for Member C Modules...\n")
    # test_imitation_learning()
    # test_reward_pruning()
    # test_evolutionary_loop()
    test_imitation_correctness()
    test_pruning_correctness()
    test_evolution_correctness()
    print("\nTesting finished.")