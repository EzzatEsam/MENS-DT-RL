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
import io
from contextlib import redirect_stdout
from training import train_mens_dt_rl
from tree_node import print_tree, tree_to_mermaid


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
        default=40,
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

    # Testing / Inference Settings
    parser.add_argument(
        "--test", action="store_true", help="Run a saved model in inference mode"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to the model to test"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["tree", "expert"],
        default="tree",
        help="Type of model to test: 'tree' (pickled DecisionTreeModel) or 'expert' (deep model)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
        help="Maximum depth of the decision trees",
    )

    return parser.parse_args()


def save_results(
    best_tree,
    best_scores,
    avg_scores,
    base_path,
    env_name,
    init_mode,
    action_names=None,
    decimals: int = 3,
):
    """
    Saves the execution results including CSV histories, plots, and
    the best decision tree pickled instance.
    """
    output_dir = os.path.dirname(base_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Save CSV
    df = pd.DataFrame(
        {
            "Generation": range(1, len(best_scores) + 1),
            "Best_Fitness": best_scores,
            "Average_Fitness": avg_scores,
        }
    )
    csv_path = f"{base_path}_history.csv"
    df.to_csv(csv_path, index=False)

    # 2. Save Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["Generation"],
        df["Best_Fitness"],
        label="Best Fitness",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        df["Generation"],
        df["Average_Fitness"],
        label="Average Fitness",
        color="orange",
        linewidth=2,
    )
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

    # 4. Save Tree Text
    tree_text_path = f"{base_path}_tree.txt"
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_tree(best_tree.root, action_names=action_names, decimals=decimals)
    with open(tree_text_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    # 5. Save Mermaid
    mermaid_path = f"{base_path}_tree.mmd"
    mermaid_text = tree_to_mermaid(
        best_tree.root, action_names=action_names, decimals=decimals
    )
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_text)

    mermaid_md_path = f"{base_path}_tree.md"
    with open(mermaid_md_path, "w", encoding="utf-8") as f:
        f.write("```mermaid\n")
        f.write(mermaid_text)
        f.write("\n```")

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
    if args.test:
        if not args.model_path:
            print("Error: --test mode requires --model_path to be specified.")
            return

        env = gym.make(args.env, render_mode="human")
        print(
            f"\n[Test Mode] Environment '{args.env}' successfully loaded with render_mode='human'."
        )

        if args.model_type == "tree":
            with open(args.model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded Decision Tree from {args.model_path}")

            from evaluation import simulate_episode

            print("Running 5 test episodes...")
            for i in range(5):
                reward = simulate_episode(model, env, render=True)
                print(f" Episode {i+1}: Reward = {reward}")
        else:
            # Expert model testing
            from expert_model import load_expert_from_path

            model = load_expert_from_path(args.model_path, env)
            print(f"Loaded Expert Model from {args.model_path}")

            print("Running 5 test episodes...")
            for i in range(5):
                obs, _ = env.reset()
                total_reward = 0.0
                done = False
                while not done:
                    env.render()
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                print(f" Episode {i+1}: Reward = {total_reward}")

        env.close()
        return

    env = gym.make(args.env)
    print(f"\n[1/4] Environment '{args.env}' successfully loaded.")

    # Generate timestamp and base path here so they are fixed from the start
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{args.env}_{args.init_mode}_{timestamp}"
    base_path = os.path.join(args.output_dir, base_filename)

    # 4. Starting the training process
    print(f"[2/4] Initializing and training population using mode: {args.init_mode}...")

    def save_callback(best_tree, best_hist, avg_hist):
        save_results(
            best_tree, best_hist, avg_hist, base_path, args.env, args.init_mode
        )

    best_tree, best_hist, avg_hist = train_mens_dt_rl(
        env=env,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        n_episodes=args.n_episodes,
        alpha=args.alpha,
        init_mode=args.init_mode,
        expert_path=args.expert_path,
        dagger_iterations=args.dagger_iterations,
        save_callback=save_callback,
        max_depth=args.max_depth,
    )

    # 5. Export and evaluation
    print(
        f"\n[3/4] Training complete. Evolution converged on a tree with fitness: {best_tree.get_fitness():.4f}"
    )

    # Final save is already handled by callback on last generation, but let's ensure it runs
    save_results(
        best_tree=best_tree,
        best_scores=best_hist,
        avg_scores=avg_hist,
        base_path=base_path,
        env_name=args.env,
        init_mode=args.init_mode,
    )

    print(f"[4/4] Results exported successfully!")
    print(f"      - History: {base_path}_history.csv")
    print(f"      - Plot:    {base_path}_plot.png")
    print(f"      - Tree:    {base_path}_tree.pkl")
    print(f"      - Tree TXT:{base_path}_tree.txt")
    print(f"      - Mermaid:{base_path}_tree.mmd")
    print(f"      - Mermaid MD:{base_path}_tree.md")

    env.close()


if __name__ == "__main__":
    main()
