"""Test recurrent functionnalities of the network"""

from typing import Sequence

import jax
import jax.numpy as jnp

from jaxppo.networks.networks_RNN import (
    NetworkRNN,
    ScannedRNN,
    get_adam_tx,
    get_model_and_state,
    init_hidden_state,
)


def test_init_and_forward_pass_network_RNN():
    """Check that the init and  forward pass doesn't fail"""
    obs_shape = (4,)
    num_envs = 4
    init_x = (
        jnp.ones((1, num_envs, *obs_shape), dtype=jnp.float32),
        jnp.ones((1, num_envs), dtype=jnp.float32),
    )
    input_architecture = ["32", "tanh", "8"]
    network = NetworkRNN(input_architecture, actor=False)
    rng = jax.random.PRNGKey(42)
    hidden_state = init_hidden_state(64, num_envs=num_envs, rng=rng)
    variables = network.init(rng, hidden_state, init_x)
    new_hidden_state, _ = network.apply(variables, hidden_state, init_x)
    assert not jnp.array_equal(hidden_state, new_hidden_state)


import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict


def params_equal(params1, params2, atol=1e-6):
    """Check if two sets of Flax parameters are equal within a tolerance."""
    # Flatten the dictionaries for easier comparison
    flat1 = flatten_dict(unfreeze(params1))
    flat2 = flatten_dict(unfreeze(params2))

    if flat1.keys() != flat2.keys():
        print("Parameter structures do not match.")
        return False

    for key in flat1:
        if not jnp.allclose(flat1[key], flat2[key], atol=atol):
            print(f"Difference found at {key}")
            return False

    return True


def test_get_network_and_state():
    obs_shape = (4,)
    num_envs = 4
    init_x = (
        jnp.ones((1, num_envs, *obs_shape), dtype=jnp.float32),
        jnp.ones((1, num_envs), dtype=jnp.float32),
    )
    input_architecture = ["32", "tanh", "8"]
    network = NetworkRNN(input_architecture, actor=False)
    tx = get_adam_tx()
    rng = jax.random.PRNGKey(42)
    model, hidden = get_model_and_state(
        network=network,
        key=rng,
        lstm_hidden_size=64,
        num_envs=num_envs,
        tx=tx,
        init_x=init_x,
    )
    assert not isinstance(model, Sequence)
    assert not isinstance(hidden, Sequence)

    networks = [NetworkRNN(input_architecture, actor=False) for _ in range(2)]
    model, hidden = get_model_and_state(
        network=networks,
        key=rng,
        lstm_hidden_size=64,
        num_envs=num_envs,
        tx=tx,
        init_x=init_x,
    )
    assert len(model) == 2
    assert len(hidden) == 2

    assert not params_equal(model[0].params, model[1].params)
