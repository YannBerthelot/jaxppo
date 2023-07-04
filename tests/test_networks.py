from jaxppo.networks import Network
import jax
import pytest
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Union, Callable
import pdb
from flax.core import freeze, unfreeze


activation_function_type = type(Callable[[jax.typing.ArrayLike], jax.Array])


def check_nn_is_equal(
    nn_1: List[Union[nn.Dense, activation_function_type]], nn_2
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
    architecture = ["64", "tanh", "32", "relu"]
    obs_dim = 2
    num_actions = 2
    return Network(
        obs_dim=obs_dim,
        input_architecture=architecture,
        actor=True,
        num_of_actions=num_actions,
    )


@pytest.fixture
def setup_simple_critic():
    architecture = ["64", "tanh", "32", "relu"]
    obs_dim = 2
    return Network(
        obs_dim=obs_dim,
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
        obs_dim=obs_dim,
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
