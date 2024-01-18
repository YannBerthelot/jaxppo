# pylint: disable = missing-function-docstring, missing-module-docstring
from typing import TypeAlias, Union

import flax.linen as nn
import gymnax
import jax
import jaxlib
import pytest
from jax import random

from jaxppo.networks.networks import (
    NetworkClassic,
    NetworkRNN,
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
    predict_probs,
    predict_value,
)
from jaxppo.utils import get_num_actions

ActivationFunction: TypeAlias = Union[
    jax._src.custom_derivatives.custom_jvp, jaxlib.xla_extension.PjitFunction
]


def check_nn_is_equal(
    nn_1: list[nn.Dense | ActivationFunction],
    nn_2: list[nn.Dense | ActivationFunction],
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
    return NetworkClassic(
        input_architecture=architecture,
        actor=True,
        num_of_actions=num_actions,
    )


@pytest.fixture
def setup_simple_critic():
    architecture = ["64", "tanh", "32", "relu"]
    return NetworkClassic(
        input_architecture=architecture,
        actor=False,
    )


def test_actor_init(setup_simple_actor):
    """Check that the network architectures matches the requested architecture"""
    num_actions = 2
    actor = setup_simple_actor
    expected_actor = nn.Sequential([
        nn.Dense(64),
        nn.tanh,
        nn.Dense(32),
        nn.relu,
        nn.Dense(num_actions),
    ])
    check_nn_is_equal(actor.architecture.layers, expected_actor.layers)


def test_critic_init(setup_simple_critic):
    """Check that the network architectures matches the requested architecture"""
    critic = setup_simple_critic
    expected_critic = nn.Sequential([
        nn.Dense(64),
        nn.tanh,
        nn.Dense(32),
        nn.relu,
        nn.Dense(1),
    ])
    check_nn_is_equal(critic.architecture.layers, expected_critic.layers)


def test_network_init():
    env, _ = gymnax.make("CartPole-v1")
    num_actions = 2
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(
        env, actor_architecture, critic_architecture, multiple_envs=False
    )
    expected_actor = nn.Sequential([
        nn.Dense(32),
        nn.tanh,
        nn.Dense(32),
        nn.tanh,
        nn.Dense(num_actions),
    ])
    check_nn_is_equal(actor.architecture.layers, expected_actor.layers)

    expected_critic = nn.Sequential([
        nn.Dense(32),
        nn.relu,
        nn.Dense(32),
        nn.relu,
        nn.Dense(1),
    ])
    check_nn_is_equal(critic.architecture.layers, expected_critic.layers)


def test_predict_value():
    env, env_params = gymnax.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(
        env, actor_architecture, critic_architecture, multiple_envs=False
    )

    key = random.PRNGKey(42)
    actor_key, critic_key, reset_key = random.split(key, num=3)
    tx = get_adam_tx()
    (_, critic_state), _ = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        actor_tx=tx,
        critic_tx=tx,
        env_params=env_params,
    )
    obs = env.reset(reset_key, env_params)[0]
    value, _ = predict_value(
        critic_state=critic_state, critic_params=critic_state.params, obs=obs
    )
    assert isinstance(value, jax.Array)


def test_predict_probs():
    env, env_params = gymnax.make("CartPole-v1")
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(
        env, actor_architecture, critic_architecture, multiple_envs=False
    )

    key = random.PRNGKey(42)
    actor_key, critic_key, reset_key = random.split(key, num=3)
    tx = get_adam_tx()
    (actor_state, _), _ = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        actor_tx=tx,
        critic_tx=tx,
        env_params=env_params,
    )
    obs = env.reset(reset_key, env_params)[0]
    probs = predict_probs(
        actor_state=actor_state, actor_params=actor_state.params, obs=obs
    )
    assert isinstance(probs, jax.Array)
    assert len(probs) == get_num_actions(env)
