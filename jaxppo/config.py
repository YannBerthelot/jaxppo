"""Config for PPO agent"""
from typing import Sequence

from pydantic import BaseModel


class PPOConfig(BaseModel):
    """Config for PPO agent"""

    total_timesteps: int
    num_steps: int
    num_envs: int
    env_id: str
    learning_rate: float
    num_minibatches: int = 4
    update_epochs: int = 4
    actor_architecture: Sequence[str] = ["64", "tanh", "64", "tanh"]
    critic_architecture: Sequence[str] = ["64", "relu", "relu", "tanh"]
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    log: bool = False
    num_episode_test: int = 20
