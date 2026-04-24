import argparse
import random
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pickle
import os
from training import train_mens_dt_rl


def parse_arguments():
    """
    Parse command-line arguments for the MENS-DT-RL training script.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="MENS-DT-RL: Evolving Interpretable Decision Trees for Reinforcement Learning"
    )

    # Environment & System Settings
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="OpenAI Gym environment ID (e.g., CartPole-v1, LunarLander-v2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory path to save the final exported Decision Tree, plots, and CSVs.",
    )

    # MENS-DT-RL Initialization Modes
    parser.add_argument(
        "--init_mode",
        type=str,
        choices=["R", "IL", "P"],
        default="R",
        help="Initialization mode: 'R' (Random), 'IL' (Imitation Learning), 'P' (Pruning)",
    )

    # Imitation Learning / Expert parameters (Only used if init_mode is IL or P)
    parser.add_argument(
        "--expert_path",
        type=str,
        default=None,
        help="Path to a pre-trained expert model (required for IL and P modes)",
    )
    parser.add_argument(
        "--dagger_iterations",
        type=int,
        default=10,
        help="Number of DAgger iterations to run during Imitation Learning",
    )

    # Evolutionary Algorithm Hyperparameters
    parser.add_argument(
        "--pop_size",
        type=int,
        default=50,
        help="Number of decision trees in the ensemble population",
    )
    parser.add_argument(
        "--max_generations",
        type=int,
        default=100,
        help="Stopping criterion: Maximum number of generations to evolve",
    )

    # Fitness Function Hyperparameters
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="Number of episodes to run when calculating tree fitness",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Weight of the tree size penalty in the fitness calculation",
    )

    return parser.parse_args()


def save_results(best_tree, best_scores, avg_scores, output_dir, env_name, init_mode):
    """
    Saves the execution results including CSV histories, plots, and 
    the best decision tree pickled instance.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{env_name}_{init_mode}_{timestamp}"
    base_path = os.path.join(output_dir, base_filename)

    # 1. Save CSV
    df = pd.DataFrame({
        "Generation": range(1, len(best_scores) + 1),
        "Best_Fitness": best_scores,
        "Average_Fitness": avg_scores
    })
    csv_path = f"{base_path}_history.csv"
    df.to_csv(csv_path, index=False)

    # 2. Save Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["Generation"], df["Best_Fitness"], label="Best Fitness", color="blue", linewidth=2)
    plt.plot(df["Generation"], df["Average_Fitness"], label="Average Fitness", color="orange", linewidth=2)
    plt.title(f"MENS-DT-RL Fitness History ({env_name} | Mode: {init_mode})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plot_path = f"{base_path}_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # 3. Save Pickle
    pkl_path = f"{base_path}_tree.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(best_tree, f)

    return base_path


def main():
    """
    The main entry point for the MENS-DT-RL training application.

    This function initializes the environment, sets random seeds, and 
    launches the evolutionary training process.
    """
    # 1. Parse command line arguments
    args = parse_arguments()

    print("=== MENS-DT-RL Initialization ===")
    print(f"Environment: {args.env}")
    print(f"Init Mode: {args.init_mode}")
    print(f"Population Size: {args.pop_size} | Max Generations: {args.max_generations}")

    # 2. Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 3. Initialize the Reinforcement Learning Environment
    env = gym.make(args.env)
    print(f"\n[1/4] Environment '{args.env}' successfully loaded.")

    # 4. Starting the training process
    print(f"[2/4] Initializing and training population using mode: {args.init_mode}...")
    
    # Check for expert path if required
    if args.init_mode in ["IL", "P"] and not args.expert_path:
        print("Warning: Init mode IL/P requires --expert_path. Falling back to Random.")
        args.init_mode = "R"

    best_tree, best_hist, avg_hist = train_mens_dt_rl(
        env=env,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        n_episodes=args.n_episodes,
        alpha=args.alpha,
        init_mode=args.init_mode,
        expert_path=args.expert_path,
        dagger_iterations=args.dagger_iterations
    )

    # 5. Export and evaluation
    print(f"\n[3/4] Training complete. Evolution converged on a tree with fitness: {best_tree.get_fitness():.4f}")
    
    saved_path_base = save_results(
        best_tree=best_tree, 
        best_scores=best_hist, 
        avg_scores=avg_hist, 
        output_dir=args.output_dir, 
        env_name=args.env, 
        init_mode=args.init_mode
    )
    
    print(f"[4/4] Results exported successfully!")
    print(f"      - History: {saved_path_base}_history.csv")
    print(f"      - Plot:    {saved_path_base}_plot.png")
    print(f"      - Tree:    {saved_path_base}_tree.pkl")

    env.close()


if __name__ == "__main__":
    main()
