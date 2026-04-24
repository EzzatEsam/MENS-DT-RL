# 📋 MENS-DT-RL — Work Distribution

> **Team:** 3 members | **Paper:** *Evolving Interpretable Decision Trees for Reinforcement Learning* (Costa et al., 2024)  
> **Repo Structure:** `decision_tree.py` · `evaluation.py` · `expert_model.py` · `imitation.py` · `training.py`

Each member owns a coherent vertical slice of the system, from the data structure layer up through the algorithm.  
Tick a box when the function body is complete, tested, and passes a quick sanity check.

---

## 👤 Member A — Decision Tree Core & Mutation Engine
> **Files:** `decision_tree.py`  
> **Theme:** Own the tree data structure and every genetic operator that transforms it.

### Phase 1 — Tree Internals

- [ ] **`DecisionTree.__init__`** — Define the node structure (inner nodes store `feature_index`, `threshold`; leaves store `action`). Support both discrete and continuous action spaces.
- [ ] **`DecisionTree.predict(state)`** — Traverse the tree from root to leaf given a normalized state vector and return the action stored at the reached leaf.
- [ ] **`DecisionTree.get_size()`** — Recursively count and return the total number of nodes (inner + leaf).
- [ ] **`DecisionTree.clone()`** — Return a fully independent deep copy of the tree (no shared node references).
- [ ] **`DecisionTree.fit(states, actions, max_depth)`** — Wrap scikit-learn's `DecisionTreeClassifier` / `DecisionTreeRegressor` (CART) to fit the tree on a `(states, actions)` dataset. Convert the sklearn internals into the custom node representation.

### Phase 2 — Genetic Mutation Operators (7 operators from §3 of the paper)

- [ ] **`mutate` → `Replace_with_child`** — Pick a random inner node; replace it with its left *or* right subtree (chosen uniformly).
- [ ] **`mutate` → `Truncate`** — Pick a random inner node; replace its left *or* right subtree with a freshly generated random leaf.
- [ ] **`mutate` → `Insert_inner_node`** — Pick a random inner node; insert a new random inner node between it and its parent, attaching the picked node as the left or right child with 50-50 probability.
- [ ] **`mutate` → `Expand_leaf`** — Pick a random leaf; convert it into an inner node with two new random leaves as children.
- [ ] **`mutate` → `Reset_split`** — Pick a random inner node; assign it a new random feature and a new threshold sampled uniformly in `[-1, 1]`.
- [ ] **`mutate` → `Modify_threshold`** — Pick a random inner node; perturb its threshold with Gaussian noise `N(0, 0.1)`.
- [ ] **`mutate` → `Modify_leaf`** — Pick a random leaf; replace its action with a new random discrete action *or* a random continuous value in the valid action range.

### Phase 3 — Population Initialization

- [ ] **`generate_random_tree(max_depth, state_space, action_space)`** — Build a complete random tree of the given depth: every inner node gets a random feature/threshold in `[-1,1]`, every leaf gets a random valid action.
- [ ] **`initialize_population(mode, pop_size, env)`** — Dispatch to the correct initialization strategy (`'R'` → all random trees; `'IL'` → random trees + 1 imitation tree; `'P'` → random trees + 1 reward-pruned imitation tree). Calls `generate_random_tree`, `run_imitation`, and `reward_pruning` as needed.

---

## 👤 Member B — Evaluation, Fitness & Expert Model
> **Files:** `evaluation.py` · `expert_model.py`  
> **Theme:** Own everything related to measuring how good a tree is and providing the black-box expert.

### Phase 1 — Episode Simulation

- [ ] **`simulate_episode(tree, env)`** — Run one full gymnasium episode using the tree as the policy. Apply attribute normalization (clip known-bounded features to `[-1, 1]`) at each timestep. Return the total undiscounted reward `G(M, i)`.
- [ ] **`evaluate_tree_performance(tree, env, n_episodes)`** — Call `simulate_episode` `n_episodes` times and return the list of all episode rewards.

### Phase 2 — Fitness & Success Rate

- [ ] **`calculate_fitness(rewards, tree_size, alpha)`** — Implement Equation (1) from the paper:  
  `fitness = mean(rewards) − std(rewards) − alpha × tree_size`
- [ ] **`calculate_success_rate(rewards, threshold)`** — Return the fraction of episodes where the total reward exceeds the environment's success threshold (used by Reward Pruning to guard against stochastic noise).

### Phase 3 — Expert Model

- [ ] **`get_expert_model(env, use_sb3=False)`** — If `use_sb3=True`, load or train a Stable-Baselines3 model (DQN for discrete, PPO for continuous). If `use_sb3=False`, define and train a custom PyTorch DNN (2 hidden layers × 32 neurons, ReLU activations) with experience-replay Q-learning. Return the trained model.
- [ ] **`load_expert_from_path(expert_path, env)`** — Load a pre-trained SB3 `.zip` model from disk using `stable_baselines3.base_class.BaseAlgorithm.load`. Handle the discrete vs. continuous action-space dispatch.

---

## 👤 Member C — Imitation Learning, Reward Pruning & Training Loop
> **Files:** `imitation.py` · `training.py`  
> **Theme:** Own the DAgger pipeline, the Reward Pruning algorithm (Algorithm 1), and the top-level evolutionary loop.

### Phase 1 — DAgger Imitation Learning

- [ ] **`run_imitation(expert_model, d_tree, env, n_episodes)`** — Implement the DAgger loop:
  1. Roll out the current student tree to collect visited states.
  2. Query the expert model to label those states with expert actions.
  3. Aggregate the new `(state, expert_action)` pairs into the dataset.
  4. Call `d_tree.fit(states, actions)` with an elbow-method regularization sweep to find the smallest tree that maintains performance.
  5. Repeat for `n_episodes` iterations; return the final fitted tree.

### Phase 2 — Reward Pruning (Algorithm 1)

- [ ] **`reward_pruning(tree, env, old_score, n_rounds, n_episodes, alpha)`** — Implement Algorithm 1 from the paper:
  - Traverse all inner nodes bottom-up, left-to-right.
  - For each node, try replacing it with its **left** child (`M'`).
  - Accept the replacement if `fitness(M') ≥ fitness(M)` **or** `success_rate(M') ≥ success_rate(M)`.
  - Else try replacing with the **right** child (`M''`) under the same condition.
  - Repeat for `n_rounds` full passes over the tree.
  - Return the pruned tree.

### Phase 3 — Evolutionary Training Loop

- [ ] **`train_mens_dt_rl(env, pop_size, max_generations, n_episodes, alpha, init_mode)`** — Complete the CRO-SL-inspired loop in `training.py`:
  - Call `initialize_population` to seed the population.
  - Each generation: clone every individual, apply a **randomly selected** mutation operator (uniform draw over all 7 operators).
  - Evaluate only un-evaluated offspring (cache fitness on already-evaluated trees).
  - Combine parents + offspring, sort by fitness descending, keep top `pop_size`.
  - Track and log `best_fitness` and `avg_fitness` per generation.
  - Return `(best_tree, best_scores, avg_scores)`.

---

## 🔗 Integration Checklist (all 3 members)

These tasks require coordination and should be completed together after individual phases are done.

- [ ] **End-to-end smoke test** — Run `python main.py --env CartPole-v1 --init_mode R --pop_size 10 --max_generations 5` without errors.
- [ ] **IL pipeline integration test** — Run `--init_mode IL` with a freshly trained SB3 expert, confirm the DAgger tree is added to the population.
- [ ] **Pruning pipeline integration test** — Run `--init_mode P`, confirm the pruned tree is smaller than the raw IL tree before evolution starts.
- [ ] **Fitness equation verification** — Unit-test `calculate_fitness` against a hand-computed example from the paper.
- [ ] **Attribute normalization check** — Confirm `simulate_episode` normalizes CartPole's four attributes correctly before passing to `predict`.
- [ ] **Continuous action space test** — Confirm the Maize Fertilization / continuous-action path works through `Modify_leaf` and `calculate_fitness`.

---

## 📊 Task Count Summary

| Member | Functions to Implement | Approximate Complexity |
|--------|------------------------|------------------------|
| **A** | 9 functions (2 structural + 7 mutation operators) | High — tree data structure design decisions affect everyone |
| **B** | 6 functions (simulation + fitness + expert) | Medium-High — must get normalization and fitness formula exactly right |
| **C** | 3 functions (DAgger + pruning + training loop) | High — algorithms are stateful and paper-faithful precision matters |
| **All** | 6 integration tasks | Low per-member — mostly running and verifying |
