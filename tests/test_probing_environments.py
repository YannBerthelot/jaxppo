r"""Test the agent on simple environments that should help pointing out which part of the\
 agent may be failing"""

from typing import Any, List

import gymnasium as gym
import numpy as np
from probing_environments.checks import (
    check_actor_and_critic_coupling,
    check_advantage_policy,
    check_backprop_value_net,
    check_loss_or_optimizer_value_net,
    check_reward_discounting,
)
from probing_environments.utils.type_hints import AgentType

from jaxppo.ppo import PPO
from jaxppo.utils import make_envs
from jaxppo.wandb_logging import LoggingConfig


def init_agent(
    agent: type[PPO],
    env: gym.Env,
    run_name: str,
    gamma: float = 0.5,
    learning_rate: float = 1e-3,
    num_envs: int = 1,
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
    envs = make_envs(
        env(num_envs, sequential=True),  # type: ignore
        capture_video=False,
        num_envs=num_envs,
    )
    return agent(  # type: ignore
        seed=42,
        num_envs=2,
        num_steps=2,
        env=envs,
        gamma=gamma,
        gae_lambda=1,
        learning_rate=learning_rate,
        entropy_coef=0.1,
        logging_config=LoggingConfig(
            project_name="debug jaxppo",
            run_name=run_name,
            config={"test": "test"},
        ),
    )


def train_agent(agent: PPO, budget: int = int(1e3)) -> AgentType:
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
    agent.train(agent.env, int(budget))
    return agent


def get_value(agent: type[PPO], obs: np.ndarray) -> np.ndarray:
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
    return agent.get_value(obs)  # type: ignore


def get_policy(agent: type[PPO], obs: np.ndarray) -> List[float]:
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
    return agent.get_probs(obs)  # type: ignore


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
    return agent.gamma


AgentType = Any
AGENT = PPO
LEARNING_RATE = 1e-2
BUDGET = 1e2


def test_check_loss_or_optimizer_value_net():
    """
    Test that check_loss_or_optimizer_value_net works on failproof sb3.
    """
    check_loss_or_optimizer_value_net(
        AGENT,
        init_agent,
        train_agent,
        get_value,
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        num_envs=2,
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
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        num_envs=2,
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
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        num_envs=2,
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
        num_envs=2,
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
        learning_rate=LEARNING_RATE,
        budget=BUDGET,
        num_envs=2,
    )


# def test_errors():
#     """
#     Run all tests on sb3 (that we assume to work) and make sure they pass and don't\
#           return any bugs.
#     """
#     with pytest.raises(AssertionError):
#         check_loss_or_optimizer_value_net(
#             AGENT, init_agent, train_agent, get_value=lambda x, y: -1
#         )
#     with pytest.raises(AssertionError):
#         check_backprop_value_net(
#             AGENT, init_agent, train_agent, get_value=lambda x, y: -1
#         )
#     with pytest.raises(AssertionError):
#         check_reward_discounting(
#             AGENT,
#             init_agent,
#             train_agent,
#             get_value=lambda x, y: -1,
#             get_gamma=get_gamma,
#         )

#     with pytest.raises(AssertionError):
#         check_advantage_policy(
#             AGENT,
#             init_agent,
#             train_agent,
#             get_policy=lambda x, y: [0.1, 0.9],
#         )

#     with pytest.raises(AssertionError):
#         check_actor_and_critic_coupling(
#             AGENT,
#             init_agent,
#             train_agent,
#             get_policy=lambda x, y: [0.1, 0.9],
#             get_value=lambda x, y: -1,
#         )
