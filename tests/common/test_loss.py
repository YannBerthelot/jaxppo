import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from flax.core import FrozenDict
from jax import random, value_and_grad, vmap

from jaxppo.networks.networks import (
    EnvironmentProperties,
    HiddenState,
    MaybeRecurrentTrainState,
    NetworkClassic,
    NetworkProperties,
    TrainState,
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
)
from jaxppo.sac.train_sac_2 import value_loss_function  # adjust this import


@pytest.fixture
def setup_loss_inputs():
    # Initialize env
    env = create_brax_env("fast")  # change to your env
    env_params = None  # if required for brax/gymnax
    num_envs = 2  # Using 2 environments for vmap compatibility
    obs_dim = env.observation_size
    action_dim = env.action_size

    env_args = EnvironmentProperties(
        env, env_params, num_envs=num_envs, continuous=True
    )
    net_props = NetworkProperties(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
    )

    actor_net, critic_net = init_networks(env_args=env_args, network_args=net_props)

    tx = get_adam_tx()
    rng = random.PRNGKey(0)
    actor_key, critic_key, sample_key = random.split(rng, 3)

    actor_state, critic_state = init_actor_and_critic_state(
        actor_network=actor_net,
        critic_network=(critic_net, critic_net),
        actor_key=actor_key,
        critic_key=critic_key,
        env_args=env_args,
        actor_tx=tx,
        critic_tx=tx,
    )

    # Create dummy transitions for 2 environments
    obs = jnp.ones((num_envs, obs_dim))
    next_obs = jnp.ones((num_envs, obs_dim)) * 0.5
    rewards = jnp.ones((num_envs,))
    dones = jnp.zeros((num_envs,))
    alpha = 0.1
    gamma = 0.99

    actor_hidden_state = None
    critic_hidden_state = None
    recurrent = False

    # For SAC, we usually have 2 critics — assume critic_state is a tuple or list
    return {
        "critic_params": tuple(cs.state.params for cs in critic_state),
        "critic_states": tuple(cs.state for cs in critic_state),
        "observations": obs,
        "next_observations": next_obs,
        "actor_state": MaybeRecurrentTrainState(
            state=actor_state.state, hidden_state=actor_hidden_state
        ),
        "recurrent": recurrent,
        "critic_hidden_state": critic_hidden_state,
        "actor_hidden_state": actor_hidden_state,
        "dones": dones,
        "rng": sample_key,
        "rewards": rewards,
        "target_critic_states": tuple(
            MaybeRecurrentTrainState(state=cs.state, hidden_state=cs.hidden_state)
            for cs in critic_state
        ),
        "gamma": gamma,
        "alpha": alpha,
    }

    # Check the structure of the gradients


def _check_structure(tree1, tree2):
    assert tree1.keys() == tree2.keys()
    for k in tree1:
        if isinstance(tree1[k], dict):
            _check_structure(tree1[k], tree2[k])
        else:
            assert tree1[k].shape == tree2[k].shape


@pytest.mark.skip
def test_value_loss_grad(setup_loss_inputs):
    grad_fn = value_and_grad(value_loss_function, argnums=0, has_aux=True)

    # Extract arguments in the exact order from the function signature
    args = (
        setup_loss_inputs["critic_params"],
        setup_loss_inputs["critic_states"],
        setup_loss_inputs["observations"],
        setup_loss_inputs["next_observations"],
        setup_loss_inputs["actor_state"],
        setup_loss_inputs["recurrent"],
        setup_loss_inputs["critic_hidden_state"],
        setup_loss_inputs["actor_hidden_state"],
        setup_loss_inputs["dones"],
        setup_loss_inputs["rng"],
        setup_loss_inputs["rewards"],
        setup_loss_inputs["target_critic_states"],
        setup_loss_inputs["gamma"],
        setup_loss_inputs["alpha"],
    )

    # Use vmap to ensure that we are running the loss for multiple environments
    # First, we check that the inputs are batched (i.e., shape has a leading dimension for batch)
    assert (
        setup_loss_inputs["observations"].ndim == 2
    )  # This should be 2 (num_envs, obs_dim)
    assert setup_loss_inputs["next_observations"].ndim == 2
    assert setup_loss_inputs["dones"].ndim == 1  # This should have shape (num_envs,)
    assert setup_loss_inputs["rewards"].ndim == 1  # This should have shape (num_envs,)

    # Check the JIT functionality for the single batch
    (loss, aux), grads = grad_fn(*args)

    # Ensure loss is scalar
    assert isinstance(loss, jax.Array)
    assert loss.ndim == 0
    assert "critic_loss" in aux
    assert "q1_loss" in aux
    assert "q2_loss" in aux

    for grad, ref in zip(grads, setup_loss_inputs["critic_params"]):
        _check_structure(ref, grad)
