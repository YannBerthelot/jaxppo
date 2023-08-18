import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxppo.networks.networks_lstm import NetworkLSTM, init_hidden_state, ScannedRNN


# def check_hidden_state_equal(hs_1, hs_2):
#     return jnp.array_equal(hs_1.h, hs_2.h) & jnp.array_equal(hs_1.c, hs_2.c)


def test_init_and_forward_pass_network_LSTM():
    """Check that the init and  forward pass doesn't fail"""
    obs_shape = (4,)
    num_envs = 4
    init_x = (
        jnp.ones((1, num_envs, *obs_shape), dtype=jnp.float32),
        jnp.ones((1, num_envs), dtype=jnp.float32),
    )
    input_architecture = ["32", "tanh", "8"]
    network = NetworkLSTM(input_architecture, actor=False)
    rng = jax.random.PRNGKey(42)
    hidden_state = init_hidden_state(
        layer=ScannedRNN(64), num_envs=num_envs, rng=rng
    )  # TODO : add hidden_size param/attribute into network to make it coherent everywhere
    variables = network.init(rng, hidden_state, init_x)
    new_hidden_state, val = network.apply(variables, hidden_state, init_x)
    assert not jnp.array_equal(hidden_state, new_hidden_state)
