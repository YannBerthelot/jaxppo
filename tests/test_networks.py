# pylint: disable = missing-function-docstring, missing-module-docstring
from typing import List, TypeAlias, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import jaxlib
import pytest
from flax.core import freeze, unfreeze
from jax import random

from jaxppo.networks import (
    Network,
    get_adam_tx,
    init_agent_state,
    init_networks,
    predict_action_logits,
    predict_value,
)
from jaxppo.utils import get_num_actions, make_envs

ActivationFunction: TypeAlias = Union[
    jax._src.custom_derivatives.custom_jvp, jaxlib.xla_extension.PjitFunction
]


def check_nn_is_equal(
    nn_1: List[Union[nn.Dense, ActivationFunction]],
    nn_2: List[Union[nn.Dense, ActivationFunction]],
) -> None:
    """
    Check if layers are equivalent:
    - if it's an activation, check that it's the same type
    - if it's a dense, check it has the same number of neurons
    """
    for cell_1, cell_2 in zip(nn_1, nn_2):
        assert isinstance(cell_1, type(cell_2))
        if isinstance(cell_1, nn.linear.Dense):
            assert cell_1.features == cell_2.features
        else:
            assert cell_1 == cell_2


@pytest.fixture
def setup_simple_actor():
    architecture = ["64", nn.tanh, "32", "relu"]
    num_actions = 2
    return Network(
        input_architecture=architecture,
        actor=True,
        num_of_actions=num_actions,
    )


@pytest.fixture
def setup_simple_critic():
    architecture = ["64", "tanh", "32", "relu"]
    return Network(
        input_architecture=architecture,
        actor=False,
    )


def test_actor_init(setup_simple_actor):
    """Check that the network architectures matches the requested architecture"""
    num_actions = 2
    actor = setup_simple_actor
    expected_actor = nn.Sequential(
        [
            nn.Dense(64),
            nn.tanh,
            nn.Dense(32),
            nn.relu,
            nn.Dense(num_actions),
        ]
    )
    check_nn_is_equal(actor.architecture.layers, expected_actor.layers)


def test_critic_init(setup_simple_critic):
    """Check that the network architectures matches the requested architecture"""

    critic = setup_simple_critic
    expected_critic = nn.Sequential(
        [
            nn.Dense(64),
            nn.tanh,
            nn.Dense(32),
            nn.relu,
            nn.Dense(1),
        ]
    )
    check_nn_is_equal(critic.architecture.layers, expected_critic.layers)


def test_network_forward_pass():
    """Check that the forward pass matches with hand computation on a simple net"""
    # Simple network to run calculations by hand
    architecture = ["2", "relu"]
    obs_dim = 2
    num_actions = 1
    actor = Network(
        input_architecture=architecture,
        actor=True,
        num_of_actions=num_actions,
    )
    key = jax.random.PRNGKey(42)
    obs = jnp.ones((1, obs_dim))
    variables = actor.init(key, obs)

    # Change weights for something we can compute by hand
    x_1_1, x_1_2, x_2_1, x_2_2 = 0.1, -0.3, 0.5, 0.6
    y_1, y_2 = -0.7, 0.2
    variables = unfreeze(variables)
    variables["params"]["Dense_0"]["kernel"] = jnp.array(
        [[x_1_1, x_1_2], [x_2_1, x_2_2]]
    )
    variables["params"]["Dense_1"]["kernel"] = jnp.array([[y_1], [y_2]])
    variables = freeze(variables)

    expected_output = jnp.array(
        max(0, (x_1_1 + x_2_1)) * (y_1) + max(0, (x_1_2 + x_2_2)) * (y_2)
    )
    output = actor.apply(variables, obs)
    assert jnp.isclose(output, expected_output)


def test_RNG_init(setup_simple_actor):
    """Check that RNG gives the same results with the same key"""

    actor = setup_simple_actor
    obs_dim = 2
    obs = jnp.ones((1, obs_dim))
    key = jax.random.PRNGKey(42)
    variables = actor.init(key, obs)
    output = actor.apply(variables, obs)
    del actor, variables

    actor = setup_simple_actor
    obs = jnp.ones((1, obs_dim))
    key = jax.random.PRNGKey(42)
    variables = actor.init(key, obs)
    new_output = actor.apply(variables, obs)

    assert jnp.array_equal(output, new_output)


def test_network_init():
    env = gym.make("CartPole-v1")
    num_actions = 2
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(env, actor_architecture, critic_architecture)
    expected_actor = nn.Sequential(
        [
            nn.Dense(32),
            nn.tanh,
            nn.Dense(32),
            nn.tanh,
            nn.Dense(num_actions),
        ]
    )
    check_nn_is_equal(actor.architecture.layers, expected_actor.layers)

    expected_critic = nn.Sequential(
        [
            nn.Dense(32),
            nn.relu,
            nn.Dense(32),
            nn.relu,
            nn.Dense(1),
        ]
    )
    check_nn_is_equal(critic.architecture.layers, expected_critic.layers)


@pytest.fixture
def setup_agent_state():
    env = gym.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(env, actor_architecture, critic_architecture)

    key = random.PRNGKey(42)
    actor_key, critic_key = random.split(key, num=2)
    tx = get_adam_tx()
    agent_state = init_agent_state(
        actor=actor,
        critic=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        tx=tx,
    )
    return agent_state


def test_init_agent_state(setup_agent_state):
    env = gym.make("CartPole-v1")
    obs = env.reset()[0]
    agent_state = setup_agent_state
    action_logits = agent_state.actor_fn(agent_state.params.actor_params, obs)
    value = agent_state.critic_fn(agent_state.params.critic_params, obs)

    assert len(action_logits) == get_num_actions(env)
    assert len(value) == 1


def test_predict_value():
    env = gym.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(env, actor_architecture, critic_architecture)

    key = random.PRNGKey(42)
    actor_key, critic_key = random.split(key, num=2)
    tx = get_adam_tx()
    agent_state = init_agent_state(
        actor=actor,
        critic=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        tx=tx,
    )
    obs = env.reset()[0]
    value = predict_value(
        agent_state=agent_state, agent_params=agent_state.params, obs=obs
    )
    assert isinstance(value, jax.Array)


def test_predict_action_logits():
    env = gym.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(env, actor_architecture, critic_architecture)

    key = random.PRNGKey(42)
    actor_key, critic_key = random.split(key, num=2)
    tx = get_adam_tx()
    agent_state = init_agent_state(
        actor=actor,
        critic=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        tx=tx,
    )
    obs = env.reset()[0]
    logits = predict_action_logits(
        agent_state=agent_state, agent_params=agent_state.params, obs=obs
    )
    assert isinstance(logits, jax.Array)
    assert len(logits) == get_num_actions(env)
