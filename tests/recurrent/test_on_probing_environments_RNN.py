"""
Premade connectors for stable-baselines3
"""
from functools import partial

# pylint: disable=W0613
from typing import Any, List, Optional

import jax
import numpy as np
import pytest
from flax import struct
from gymnax.environments.environment import Environment
from probing_environments.checks import (
    check_actor_and_critic_coupling,
    check_advantage_policy,
    check_backprop_value_net,
    check_loss_or_optimizer_value_net,
    check_recurrent_agent,
    check_reward_discounting,
)
from probing_environments.utils.type_hints import AgentType

from jaxppo.networks.networks_RNN import init_hidden_state as _init_hidden_state
from jaxppo.networks.utils import ScannedRNN
from jaxppo.ppo import PPO as classical_PPO
from jaxppo.ppo_rnn import PPO


@struct.dataclass
class EnvParams:
    """Environment parameters (unused here)"""

    unused: Optional[Any] = None


def init_agent(
    agent: AgentType,
    env: Environment,
    run_name: str,  # pylint: disable=W0613
    gamma: float = 0.5,
    learning_rate: float = 1e-3,
    num_envs: int = 4,
    seed: Optional[int] = 42,  # pylint: disable=W0613
    budget: int = int(1e3),
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
        num_steps=32,
        env_id=env(),
        env_params=env_params,
        gamma=gamma,
        learning_rate=learning_rate,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
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


def get_value(
    agent: PPO, obs: np.ndarray, done: bool = True, hidden: Optional[jax.Array] = None
) -> np.ndarray:
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
    return agent.predict_value_and_hidden(obs, done, hidden)[1].item()


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
    return agent.predict_probs(obs)  # unify with predict_value


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
LEARNING_RATE = 1e-3
BUDGET = 2e3
NUM_ENVS = 4


def test_check_check_loss_or_optimizer_value_net():
    """
    Test that check_backprop_value_net works on failproof sb3.
    """
    check_loss_or_optimizer_value_net(
        AGENT,
        partial(init_agent, budget=BUDGET),
        train_agent,
        get_value,
        num_envs=NUM_ENVS,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        gymnax=True,
    )


def test_check_backprop_value_net():
    """
    Test that check_backprop_value_net works on failproof sb3.
    """
    check_backprop_value_net(
        AGENT,
        init_agent,
        train_agent,
        get_value,
        num_envs=NUM_ENVS,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        gymnax=True,
    )


def test_check_reward_discounting():
    """
    Test that check_reward_discounting works on failproof sb3.
    """
    check_reward_discounting(
        AGENT,
        init_agent,
        train_agent,
        get_value,
        get_gamma,
        num_envs=NUM_ENVS,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        gymnax=True,
    )


def test_check_advantage_policy():
    """
    Test that check_advantage_policy works on failproof sb3.
    """
    check_advantage_policy(
        AGENT,
        init_agent,
        train_agent,
        get_policy,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        gymnax=True,
        num_envs=NUM_ENVS,
    )


def test_check_actor_and_critic_coupling():
    """
    Test that check_actor_and_critic_coupling works on failproof sb3.
    """
    check_actor_and_critic_coupling(
        AGENT,
        init_agent,
        train_agent,
        get_policy,
        get_value,
        num_envs=NUM_ENVS,
        learning_rate=LEARNING_RATE,
        budget=BUDGET * 2,
        gymnax=True,
    )


LSTM_HIDDEN_SIZE = 32


def init_hidden_state():
    """Init the hidden state of the RNN layer"""
    return _init_hidden_state(ScannedRNN(LSTM_HIDDEN_SIZE), 1, jax.random.PRNGKey(0))


def compute_next_critic_hidden(
    agent: AGENT, obs: np.ndarray, done: bool, hidden: np.ndarray
) -> tuple[jax.Array, jax.Array]:
    """Get the next critic hidden state"""
    return agent.predict_value_and_hidden(obs, done, hidden)[0]


def test_check_recurrent_agent():
    """
    Test that check_actor_and_critic_coupling works on failproof sb3.
    """
    check_recurrent_agent(
        AGENT,
        init_agent,
        train_agent,
        get_value_recurrent=get_value,
        init_hidden_state=init_hidden_state,
        compute_next_critic_hidden=compute_next_critic_hidden,
        num_envs=NUM_ENVS,
        learning_rate=LEARNING_RATE,
        budget=BUDGET * 4,
        gymnax=True,
    )


def classical_init_agent(
    agent: AgentType,
    env: Environment,
    run_name: str,  # pylint: disable=W0613
    gamma: float = 0.5,
    learning_rate: float = 1e-3,
    num_envs: int = 1,
    seed: Optional[int] = 42,  # pylint: disable=W0613
    budget: int = int(1e3),
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
    agent = classical_PPO(
        total_timesteps=budget,
        num_envs=num_envs,
        num_steps=32,
        env_id=env(),
        env_params=env_params,
        gamma=gamma,
        learning_rate=learning_rate,
    )
    return agent


def classical_get_value(
    agent: classical_PPO, obs: np.ndarray, done: Any = None, hidden: Any = None
) -> np.ndarray:
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
    return agent.predict_value(obs)


def classical_compute_next_critic_hidden(
    agent: Any = None, obs: Any = None, done: Any = None, hidden: Any = None
):
    """Fake returning a hidden state when using no RNN layer"""
    return None


def test_check_non_recurrent_agent_fails():
    """
    Test that check_actor_and_critic_coupling works on failproof sb3.
    """
    with pytest.raises(AssertionError):
        check_recurrent_agent(
            classical_PPO,
            classical_init_agent,
            train_agent,
            get_value_recurrent=classical_get_value,
            init_hidden_state=init_hidden_state,
            compute_next_critic_hidden=classical_compute_next_critic_hidden,
            num_envs=NUM_ENVS,
            learning_rate=LEARNING_RATE,
            budget=BUDGET * 4,
            gymnax=True,
        )
