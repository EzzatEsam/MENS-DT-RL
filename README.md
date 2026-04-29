# MENS-DT-RL: Evolving Interpretable Decision Trees for RL

This repository is an implementation of solving complex Reinforcement Learning tasks using evolved transparent univariate Decision Trees, based on a multi-method evolutionary ensemble strategy (MENS-DT-RL).

Unlike rigid Neural Networks, these models are deeply interpretable and easily audited, yielding transparent rules for environments ranging from Lunar Lander physics to continuous crop management strategies.

## Requirements

Ensure you are using Python 3.10+ in a virtual environment. We recommend using `uv`:

```bash
# Setup the environment and install dependencies
uv venv
uv sync
```

## Running the Training Algorithm

The entry point for this system is `main.py`. You can utilize the underlying evolutionary strategies via the CLI:

### 1. Basic Initialized Training (Random Trees)
The simplest way to initiate evolution if you do not have a trained Expert Neural Network:
```bash
python main.py --env CartPole-v1 --init_mode R --pop_size 50 --max_generations 100
```

### 2. Training with Imitation Learning (IL Mode)
This method spins up a fast, imperfect Decision Tree via the DAgger algorithm by mimicking a black-box expert, giving the Evolution a huge mathematical headstart.
```bash
# You must provide the path to your stable-baselines3 expert
python main.py --env CartPole-v1 --init_mode IL --expert_path "models/expert_cartpole.zip"
```

### 3. Training with Pruned Imitation Learning (P Mode)
This executes **Algorithm 1**, actively pruning the Imitation Learning tree directly against the RL environment to heavily shrink its node footprint *before* evolution even starts, strictly maximizing the `Avg Reward - Std Dev - (Alpha * Size)` equation out of the gate.
```bash
python main.py --env LunarLander-v2 --init_mode P --expert_path "models/expert_lunarlander.zip" --alpha 0.05
```

## Testing and Rendering

Once you have a saved model (either an evolved tree or a deep expert), you can run it in "inference mode" with visual rendering to observe its behavior.

### 1. Test a Saved Decision Tree
This will load a pickled tree, apply the required Attribute Normalization, and render the environment:
```bash
python main.py --test --env CartPole-v1 --model_path "results/CartPole-v1_P_20260429_tree.pkl" --model_type tree
```

### 2. Test a Deep Expert Model
```bash
python main.py --test --env LunarLander-v2 --model_path "models/expert_lunarlander.zip" --model_type expert
```

## Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--env` | str | `CartPole-v1` | Gymnasium environment name. |
| `--seed` | int | `42` | Evolution and environment fixed seed. |
| `--output_dir` | str | `./results` | Directory where the final evolved tree, CSV history, and plot is saved. |
| `--init_mode` | str | `R` | Evolution start point: `R` (Random), `IL` (Imitation Learning), `P` (Pruning). |
| `--expert_path` | str | `None` | The SB3 Zip required for IL or P modes. |
| `--dagger_iterations` | int | `10` | The depth of data aggregation for Imitation. |
| `--pop_size` | int | `50` | The number of competing trees per generation. |
| `--max_generations` | int | `100` | Evolutionary stopping condition. |
| `--n_episodes` | int | `100` | RL evaluation accuracy baseline. |
| `--alpha` | float | `0.01` | Regularization weight punishing bloated trees. |
| `--test` | flag | `False` | Run the specified model in inference mode with rendering. |
| `--model_path` | str | `None` | Path to the model file to test (required for `--test`). |
| `--model_type` | str | `tree` | Type of model being tested: `tree` or `expert`. |

## Core Components

- **`normalizer.py`**: Contains the `ObservationNormalizer` class which handles the paper's Attribute Normalization procedure (§3). It scales finite-bounded observations to `[-1, 1]` and supports both single observations and batch processing.
- **`evaluation.py`**: Handles simulation episodes, fitness calculation (Equation 1), and success rate metrics.
- **`decision_tree_model.py`**: The underlying mutable decision tree architecture.

## Development Tasks

The project is currently tracking progress across a 3-man development team for the underlying graph architecture, DAgger pipeline implementations, and the 7 isolated genetic operations required by the paper. 

To view the breakdown and implementation checklist, refer to [`work_distribution.md`](work_distribution.md).
