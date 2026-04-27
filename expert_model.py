import random
import collections
from typing import Any

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import Env


# ---------------------------------------------------------------------------
# Custom DQN expert (paper §4.5)
# "dense neural networks with 2 hidden layers of 32 nodes each,
#  trained with deep Q-Learning and experience replay"
# ---------------------------------------------------------------------------

class _DQNetwork(nn.Module):
    """
    Two-hidden-layer feedforward Q-network with ReLU activations.

    Architecture (paper §4.5):
        obs_dim → 32 (ReLU) → 32 (ReLU) → n_actions
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class CustomDQNExpert:
    """
    Wraps a trained _DQNetwork and exposes a .predict() interface that mirrors
    the Stable-Baselines3 convention used elsewhere in the pipeline.

    Parameters
    ----------
    obs_dim : int
    n_actions : int
    device : torch.device
    """

    def __init__(self, obs_dim: int, n_actions: int, device: T.device) -> None:
        self.device = device
        self.n_actions = n_actions
        self.q_net = _DQNetwork(obs_dim, n_actions).to(device)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """
        Return the greedy action for a given observation.

        Returns
        -------
        tuple[int, None]   — (action, state) matching SB3's predict() signature
        """
        obs_t = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0)
        with T.no_grad():
            q_values = self.q_net(obs_t)
        action = int(q_values.argmax(dim=1).item())
        return action, None


def _train_custom_dqn(
    env: Env,
    n_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 10_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.995,
    target_update_freq: int = 10,
) -> CustomDQNExpert:
    """
    Train a custom DQN expert with experience replay.

    Supports only **discrete** action spaces (Gymnasium Discrete).
    This matches the paper's use of DQN for CartPole, MountainCar, and
    LunarLander (§4.5).

    Parameters
    ----------
    env : Env
        A Gymnasium environment with a Discrete action space.
    n_episodes : int
        Number of training episodes.
    gamma : float
        Discount factor.
    lr : float
        Adam learning rate.
    batch_size : int
        Replay-buffer sampling batch size.
    buffer_capacity : int
        Maximum size of the experience replay buffer.
    eps_start / eps_end / eps_decay : float
        Epsilon-greedy exploration schedule (multiplicative decay per episode).
    target_update_freq : int
        How many episodes between hard target-network updates.

    Returns
    -------
    CustomDQNExpert
        The trained expert model.
    """
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(
            "Custom DQN training only supports Discrete action spaces. "
            "Use use_sb3=True for continuous environments."
        )

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    expert = CustomDQNExpert(obs_dim, n_actions, device)
    target_net = _DQNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(expert.q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(expert.q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Experience replay buffer: stores (obs, action, reward, next_obs, done)
    ReplayBuffer = collections.deque(maxlen=buffer_capacity)

    epsilon = eps_start

    for episode in range(n_episodes):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        episode_done = False

        while not episode_done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action, _ = expert.predict(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            episode_done = terminated or truncated

            ReplayBuffer.append((obs, action, reward, next_obs, float(episode_done)))
            obs = next_obs

            # Only learn when enough samples are available
            if len(ReplayBuffer) >= batch_size:
                batch = random.sample(ReplayBuffer, batch_size)
                obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)

                obs_t      = T.tensor(np.array(obs_b),      dtype=T.float32, device=device)
                act_t      = T.tensor(act_b,                dtype=T.long,    device=device).unsqueeze(1)
                rew_t      = T.tensor(rew_b,                dtype=T.float32, device=device).unsqueeze(1)
                next_obs_t = T.tensor(np.array(next_obs_b), dtype=T.float32, device=device)
                done_t     = T.tensor(done_b,               dtype=T.float32, device=device).unsqueeze(1)

                # Current Q-values for chosen actions
                q_current = expert.q_net(obs_t).gather(1, act_t)

                # Bellman target using frozen target network
                with T.no_grad():
                    q_next = target_net(next_obs_t).max(dim=1, keepdim=True).values
                    q_target = rew_t + gamma * q_next * (1.0 - done_t)

                loss = loss_fn(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(eps_end, epsilon * eps_decay)

        # Periodically sync target network
        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(expert.q_net.state_dict())

    return expert


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_expert_model(env: Env, use_sb3: bool = False, *args, **kwargs) -> Any:
    """
    Build and train (or load) an expert uninterpretable model for Imitation Learning.

    Two backends are available:

    Stable-Baselines3 backend (use_sb3=True)
    ------------------------------------------
    Automatically selects the appropriate algorithm:
      - Discrete action space  → DQN
      - Continuous action space → PPO  (as used for Maize Fertilization, §4.5)

    Relevant kwargs:
      total_timesteps (int)  : training budget, default 100_000
      verbose (int)          : SB3 verbosity level, default 0

    Custom PyTorch backend (use_sb3=False) — default
    --------------------------------------------------
    Implements the architecture from §4.5:
      - 2 hidden layers × 32 neurons, ReLU activations
      - Deep Q-Learning with experience replay
      - Only supports Discrete action spaces

    Relevant kwargs:
      n_episodes (int)         : training episodes,       default 500
      gamma (float)            : discount factor,          default 0.99
      lr (float)               : learning rate,            default 1e-3
      batch_size (int)         : replay batch size,        default 64
      buffer_capacity (int)    : replay buffer size,       default 10_000
      eps_start (float)        : initial epsilon,          default 1.0
      eps_end (float)          : minimum epsilon,          default 0.05
      eps_decay (float)        : per-episode decay factor, default 0.995
      target_update_freq (int) : episodes between target syncs, default 10

    Parameters
    ----------
    env : Env
        The gymnasium environment the expert will be trained on.
    use_sb3 : bool, optional
        Whether to use Stable-Baselines3. Default is False (custom DQN).

    Returns
    -------
    Any
        The trained expert model. Exposes a `.predict(obs)` method that
        returns (action, state) — compatible with the imitation.py pipeline.
    """
    if use_sb3:
        # Late import so that SB3 is only required when explicitly requested.
        try:
            from stable_baselines3 import DQN, PPO
        except ImportError as e:
            raise ImportError(
                "stable_baselines3 is not installed. "
                "Install it with: pip install stable-baselines3"
            ) from e

        total_timesteps = kwargs.get("total_timesteps", 100_000)
        verbose = kwargs.get("verbose", 0)

        if isinstance(env.action_space, gym.spaces.Discrete):
            model = DQN("MlpPolicy", env, verbose=verbose)
        else:
            model = PPO("MlpPolicy", env, verbose=verbose)

        model.learn(total_timesteps=total_timesteps)
        return model

    else:
        # Custom PyTorch DQN — §4.5
        return _train_custom_dqn(
            env=env,
            n_episodes=kwargs.get("n_episodes", 500),
            gamma=kwargs.get("gamma", 0.99),
            lr=kwargs.get("lr", 1e-3),
            batch_size=kwargs.get("batch_size", 64),
            buffer_capacity=kwargs.get("buffer_capacity", 10_000),
            eps_start=kwargs.get("eps_start", 1.0),
            eps_end=kwargs.get("eps_end", 0.05),
            eps_decay=kwargs.get("eps_decay", 0.995),
            target_update_freq=kwargs.get("target_update_freq", 10),
        )


def load_expert_from_path(expert_path: str, env: Env) -> Any:
    """
    Load a pre-trained Stable-Baselines3 model from disk.

    Dispatches to DQN.load or PPO.load based on the environment's action space:
      - Discrete action space  → DQN
      - Continuous action space → PPO

    Parameters
    ----------
    expert_path : str
        Path to the saved SB3 .zip model file.
    env : Env
        The gymnasium environment, used to determine the algorithm type.

    Returns
    -------
    Any
        The loaded SB3 model. Exposes a .predict(obs) method that returns
        (action, state) — compatible with the imitation.py pipeline.
    """
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as e:
        raise ImportError(
            "stable_baselines3 is not installed. "
            "Install it with: pip install stable-baselines3"
        ) from e

    if isinstance(env.action_space, gym.spaces.Discrete):
        model = DQN.load(expert_path, env=env)
    else:
        model = PPO.load(expert_path, env=env)

    return model
