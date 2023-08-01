# pylint: disable = missing-function-docstring, missing-module-docstring
import logging
import os
from typing import List, TypeAlias, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jaxlib
import pytest
from jax import random

from jaxppo.networks import (
    Network,
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
    predict_probs,
    predict_probs_and_value,
    predict_value,
)
from jaxppo.utils import get_num_actions

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


def test_share_layer_init():
    """Check that the network architectures matches the requested architecture"""

    architecture = ["64", "tanh", "32", "relu"]
    actor_critic = Network(
        input_architecture=architecture,
        actor=True,
        shared_network=True,
        num_of_actions=2,
    )

    expected_network = nn.Sequential(
        [
            nn.Dense(64),
            nn.tanh,
            nn.Dense(32),
            nn.relu,
        ]
    )
    check_nn_is_equal(actor_critic.architecture.layers, expected_network.layers)


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


def test_predict_value():
    env = gym.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(env, actor_architecture, critic_architecture)

    key = random.PRNGKey(42)
    actor_key, critic_key = random.split(key, num=2)
    tx = get_adam_tx()
    _, critic_state = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        tx=tx,
    )
    obs = env.reset()[0]
    value = predict_value(
        critic_state=critic_state, critic_params=critic_state.params, obs=obs
    )
    assert isinstance(value, jax.Array)


def test_predict_probs():
    env = gym.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(
        env,
        actor_architecture,
        critic_architecture,
    )

    key = random.PRNGKey(42)
    actor_key, critic_key = random.split(key, num=2)
    tx = get_adam_tx()
    actor_state, _ = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        tx=tx,
    )
    obs = env.reset()[0]
    probs = predict_probs(
        actor_state=actor_state, actor_params=actor_state.params, obs=obs
    )
    assert isinstance(probs, jax.Array)
    assert len(probs) == get_num_actions(env)


def test_shared_predict():
    env = gym.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    actor_critic, _ = init_networks(env, actor_architecture, shared_network=True)

    key = random.PRNGKey(42)
    actor_key, _ = random.split(key, num=2)
    tx = get_adam_tx()
    actor_critic_state, _ = init_actor_and_critic_state(
        actor_network=actor_critic,
        actor_key=actor_key,
        shared_network=True,
        env=env,
        tx=tx,
    )
    obs = env.reset()[0]
    probs, val = predict_probs_and_value(
        actor_critic_state=actor_critic_state,
        actor_critic_params=actor_critic_state.params,
        obs=obs,
    )
    assert isinstance(probs, jax.Array)
    assert len(probs) == get_num_actions(env)
    assert isinstance(val, jax.Array)
