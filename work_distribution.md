# MENS-DT-RL Work Distribution

Here is the tracking checklist divided across the three developers.

## Developer 1: The "Tree Architecture" Lead
**Main Files:** `decision_tree.py`

*Core Structures & Mechanics:*
- [ ] Design `SplitNode` and `LeafNode` structures capable of handling both discrete (ints) and continuous (floats) actions.
- [ ] Implement a custom recursive `fit(X, y)` function using native CART principles (Gini/MSE) to train the tree from scratch (NO sklearn).
- [ ] Implement `predict(state)` for real-time inference.
- [ ] Implement `get_size()` (counting total nodes).
- [ ] Implement `clone()` (deep copying the entire graph).

*Genetic Operators:*
- [ ] `mutate(mutation_type)` router setup.
- [ ] Implement `Replace_with_child`
- [ ] Implement `Truncate`
- [ ] Implement `Insert_inner_node`
- [ ] Implement `Expand_leaf`
- [ ] Implement `Reset_split`
- [ ] Implement `Modify_threshold`
- [ ] Implement `Modify_leaf` (checking env.action_space for continuous vs discrete randomly generated values).
- [ ] Implement `generate_random_tree(max_depth)`

*Serialization & Output:*
- [ ] Implement `save(filepath)` using pickle/json.
- [ ] Implement `load(filepath)`
- [ ] Implement text-based `print_tree()` for visual interpretability.


## Developer 2: The "RL & Evaluation" Lead
**Main Files:** `evaluation.py`, `expert_model.py`, `env_wrapper.py`

*Environments & Evaluation:*
- [ ] Implement a robust `gymnasium.ObservationWrapper` (`env_wrapper.py`) enforcing `[-1, 1]` observation bounds strictly as defined by the paper.
- [ ] Make sure wrappers correctly pass action space discrete/continuous metadata downstream.
- [ ] Implement `simulate_episode(tree, env) -> float`
- [ ] Implement `evaluate_tree_performance(tree, env, N) -> list[float]`
- [ ] Implement the `calculate_fitness()` formula: `Avg Reward - Std Dev - (Alpha * Tree Size)`.

*Expert Agents:*
- [ ] Implement `get_expert_model(env, use_sb3=True)` utilizing `stable-baselines3` (PPO or DQN) to spawn or load black-box RL models.


## Developer 3: The "Algorithmic Pipelines" Lead
**Main Files:** `imitation.py`, `training.py`, `main.py`

*Evolution & Imitation Pipelines:*
- [ ] Implement `run_imitation()` specifically running custom DAgger rollouts using Dev 2's Expert Agent.
- [ ] During `run_imitation()`, invoke Dev 1's `tree.fit(states, actions)` to generate the custom baseline Imitation Tree.
- [ ] Implement Algorithm 1 `reward_pruning()` in `training.py`. (Post-order/bottom-up traversal swapping nodes and checking if `evaluate_tree_performance` yields a higher `fitness`).
- [ ] Complete `initialize_population(mode)` handling exactly how the zero generation spawns under modes R, IL, and P.
- [ ] Ensure all input/output bindings cleanly flow through the `train_mens_dt_rl` loop in `main.py` and export the final tree correctly.
