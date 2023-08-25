"""Test recurrent functionnalities of the network"""
import jax
import jax.numpy as jnp

from jaxppo.networks.networks_RNN import NetworkRNN, ScannedRNN, init_hidden_state


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
    hidden_state = init_hidden_state(layer=ScannedRNN(64), num_envs=num_envs, rng=rng)
    variables = network.init(rng, hidden_state, init_x)
    new_hidden_state, _ = network.apply(variables, hidden_state, init_x)
    assert not jnp.array_equal(hidden_state, new_hidden_state)
