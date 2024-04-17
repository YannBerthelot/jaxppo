"""
Premade connectors for stable-baselines3
"""

from functools import partial

# pylint: disable=W0613
from typing import Any, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments.environment import Environment
from probing_environments.checks import check_average_reward
from probing_environments.utils.type_hints import AgentType

from jaxppo.ppo import PPO


@struct.dataclass
class EnvParams:
    """Environment parameters"""

    max_steps_in_episode: int = 1000


def init_agent(
    agent: AgentType,
    env: Environment,
    run_name: str,  # pylint: disable=W0613
    gamma: float = 0.5,
    learning_rate: float = 1e-3,
    num_envs: int = 1,
    seed: Optional[int] = 42,  # pylint: disable=W0613
    budget: int = int(1e3),
    continuous: bool = False,
) -> AgentType:
    """
    Initialize your agent on a given env while also setting the discount factor.

    Args:
        agent (AgentType) : The agent to be used
        env (gym.Env): The env to use with your agent.
        gamma (float, optional): The discount factor to use. Defaults to 0.5.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your agent with the right settings.
    """
    env_params = EnvParams()
    agent = PPO(
        total_timesteps=budget,
        num_envs=num_envs,
        num_steps=1024,
        env_id=env(),
        env_params=env_params,
        gamma=gamma,
        learning_rate=learning_rate,
        continuous=continuous,
        average_reward=True,
    )
    return agent


def train_agent(
    agent: PPO, budget: Optional[int] = int(1e3), seed: int = 42
) -> AgentType:
    """
    Train your agent for a given budget/number of timesteps.

    Args:
        agent (AgentType): Your agent (created by init_agent)
        budget (int, optional): The number of timesteps to train the agent on. Defaults\
              to int(1e3).

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        AgentType: Your trained agents.
    """
    agent.train(seed=seed, test=False)
    return agent


def get_value(agent: PPO, obs: np.ndarray) -> np.ndarray:
    """
    Predict the value of a given obs (in numpy array format) using your current value \
        net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        np.ndarray: The predicted value of the given observation.
    """
    return agent.predict_value(obs)[0]


def get_policy(agent: PPO, obs: np.ndarray) -> List[float]:
    """
    Predict the probabilitie of actions in a given obs (in numpy array format) using\
          your current policy net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        List[float]: The probabilities of taking every actions.
    """
    return agent.predict_probs(obs)


def get_action(agent: PPO, obs: np.ndarray, key: jax.Array) -> float:
    """
    Predict the (continuous) action using\
          your current policy net.

    Args:
        agent (AgentType): Your agent to make the prediction.
        obs (np.ndarray): The observation to make the prediction on.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        float: The action
    """

    action = agent.get_action(jnp.array(obs), key)[0]
    return action


def get_gamma(agent: PPO) -> float:
    """
    Fetch the gamma/discount factor value from your agent (to use it in tests)

    Args:
        agent (AgentType): Your agent.

    Raises:
        NotImplementedError: While you haven't implemented your own functions or picked\
              from the existing ones

    Returns:
        float: The gamma/discount factor value of your agent
    """
    return agent.config.gamma


AGENT = PPO
LEARNING_RATE = 1e-2
BUDGET = 2e5


@pytest.mark.slow
def test_check_average_reward():
    """
    Test that check_advantage_policy works on failproof sb3.
    """
    check_average_reward(
        AGENT,
        init_agent,
        train_agent,
        get_action,
        get_value,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        gymnax=True,
        key=jax.random.PRNGKey(42),
        num_envs=2,
    )
