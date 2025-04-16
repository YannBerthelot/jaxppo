# test_networks_action_value.py
# pylint: disable = missing-function-docstring, missing-module-docstring
from typing import TypeAlias, Union

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import jaxlib
import pytest
from jax import random

from jaxppo.networks.networks import (
    EnvironmentProperties,
    NetworkClassic,
    NetworkProperties,
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
    for cell_1, cell_2 in zip(nn_1, nn_2):
        assert isinstance(cell_1, type(cell_2))
        if isinstance(cell_1, nn.linear.Dense):
            assert cell_1.features == cell_2.features
        else:
            assert cell_1 == cell_2


# === Fixtures === #


@pytest.fixture
def setup_simple_action_value_critic():
    architecture = ["64", "tanh", "32", "relu"]
    return NetworkClassic(
        input_architecture=architecture,
        actor=False,
        action_value=True,  # 👈 Enable action-value mode
    )


@pytest.fixture
def setup_actor_critic_with_action_value():
    env, env_params = gymnax.make("CartPole-v1")
    num_actions = get_num_actions(env, env_params)
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    env_args = EnvironmentProperties(env, env_params, num_envs=1, continuous=False)
    network_args = NetworkProperties(
        actor_architecture, critic_architecture, action_value=True
    )
    actor, critic = init_networks(env_args=env_args, network_args=network_args)

    return (
        actor,
        critic,
        num_actions,
        env_args,
    )


@pytest.fixture
def setup_actor_critic():
    env, env_params = gymnax.make("CartPole-v1")
    num_actions = get_num_actions(env, env_params)
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    env_args = EnvironmentProperties(env, env_params, num_envs=1, continuous=False)
    network_args = NetworkProperties(actor_architecture, critic_architecture)
    actor, critic = init_networks(env_args=env_args, network_args=network_args)

    return (
        actor,
        critic,
        num_actions,
        env_args,
    )


# === Tests === #


def test_predict_value_action_value(setup_actor_critic_with_action_value):
    """Test value prediction when critic uses (obs, action) input"""
    actor, critic, num_actions, env_args = setup_actor_critic_with_action_value

    key = random.PRNGKey(42)
    actor_key, critic_key, reset_key, action_key = random.split(key, num=4)
    tx = get_adam_tx()

    actor_state, critic_state = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env_args=env_args,
        actor_tx=tx,
        critic_tx=tx,
        action_value=True,
    )

    obs = env_args.env.reset(reset_key, env_args.env_params)[0]
    action = random.randint(action_key, shape=(1,), minval=0, maxval=num_actions)

    value, _ = predict_value(
        critic_state=critic_state.state,
        critic_params=critic_state.state.params,
        obs=obs,
        action=action,  # 👈 pass action to handle action-value mode
    )
    assert isinstance(value, jax.Array)


# Original test still works for V(s) critics
def test_predict_value_state_value_only(setup_actor_critic):
    actor, critic, _, env_args = setup_actor_critic

    key = random.PRNGKey(42)
    actor_key, critic_key, reset_key = random.split(key, num=3)
    tx = get_adam_tx()
    actor_state, critic_state = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env_args=env_args,
        actor_tx=tx,
        critic_tx=tx,
    )
    obs = env_args.env.reset(reset_key, env_args.env_params)[0]
    value, _ = predict_value(
        critic_state=critic_state.state,
        critic_params=critic_state.state.params,
        obs=obs,
    )
    assert isinstance(value, jax.Array)


def test_predict_probs(setup_actor_critic):
    actor, critic, _, env_args = setup_actor_critic

    key = random.PRNGKey(42)
    actor_key, critic_key, reset_key = random.split(key, num=3)
    tx = get_adam_tx()
    actor_state, _ = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env_args=env_args,
        actor_tx=tx,
        critic_tx=tx,
    )
    obs = env_args.env.reset(reset_key, env_args.env_params)[0]
    probs = predict_probs(
        actor_state=actor_state.state, actor_params=actor_state.state.params, obs=obs
    )
    assert isinstance(probs, jax.Array)
    assert len(probs) == get_num_actions(env_args.env, env_args.env_params)
