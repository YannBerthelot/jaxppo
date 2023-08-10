import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxppo.networks.networks_lstm import NetworkLSTM, init_hidden_state, HiddenState


def check_hidden_state_equal(hs_1, hs_2):
    return jnp.array_equal(hs_1.h, hs_2.h) & jnp.array_equal(hs_1.c, hs_2.c)


def test_init_hidden_state():
    num_envs = 4
    input_architecture = ["64", "tanh", "LSTM(32)", "8"]
    network = NetworkLSTM(input_architecture, actor=False)
    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split(rng)
    hidden_state = init_hidden_state(num_envs, rng, network)
    h, c = network.extractor_architecture[2].initialize_carry(_rng, num_envs)
    expected_hidden_state = HiddenState(h=h, c=c)
    assert check_hidden_state_equal(
        hidden_state,
        expected_hidden_state,
    )


def test_init_hidden_states_LSTM_is_first_layer():
    """Check that the hidden states are initialized properly when the first layer is LSTM"""
    obs_shape = (4,)
    num_envs = 4
    init_x = (
        jnp.zeros((1, num_envs, *(obs_shape))),
        jnp.zeros((1, num_envs)),
    )
    ins, _ = init_x
    input_architecture = ["LSTM(32)", "8"]
    network = NetworkLSTM(input_architecture, actor=False)
    rng = jax.random.PRNGKey(42)

    hidden_state = init_hidden_state(num_envs, rng, network)
    rng, _rng = jax.random.split(rng)
    h, c = network.extractor_architecture[0].initialize_carry(_rng, num_envs)
    expected_hidden_state = HiddenState(h=h, c=c)
    assert check_hidden_state_equal(
        hidden_state,
        expected_hidden_state,
    )


def test_init_and_forward_pass_network_LSTM():
    """Check that the init and  forward pass doesn't fail"""
    obs_shape = (4,)
    num_envs = 4
    init_x = (
        jnp.ones((1, num_envs, *obs_shape), dtype=jnp.float32),
        jnp.ones((1, num_envs), dtype=jnp.float32),
    )
    ins, _ = init_x
    input_architecture = ["LSTM(32)", "tanh", "8"]
    network = NetworkLSTM(input_architecture, actor=False)
    rng = jax.random.PRNGKey(42)
    hidden_state = init_hidden_state(num_envs, rng, network)
    variables = network.init(rng, hidden_state, init_x)
    new_hidden_state, val = network.apply(variables, hidden_state, init_x)
    assert not check_hidden_state_equal(hidden_state, new_hidden_state)
