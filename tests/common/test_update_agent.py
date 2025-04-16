import jax
import jax.numpy as jnp
import optax
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
from jaxppo.sac.train_sac_2 import (
    AgentState,
    create_alpha_train_state,
    update_policy,
    update_temperature,
    update_value_functions,
)


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
    alpha = create_alpha_train_state()
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
        "action_dim": action_dim,
    }


@pytest.fixture
def setup_update_inputs(setup_loss_inputs):
    data = setup_loss_inputs

    agent_state = AgentState(
        rng=data["rng"],
        actor_state=data["actor_state"],
        critic_states=tuple(
            MaybeRecurrentTrainState(
                state=cs,
                hidden_state=chs,
            )
            for cs, chs in zip(data["critic_states"], data["target_critic_states"])
        ),
        target_critic_states=data["target_critic_states"],
        buffer_state=None,  # You can customize this if your update function uses buffer_state
        alpha=jnp.ones((1,)),
    )

    log_alpha = data["alpha"].params["log_alpha"]

    return {
        "observations": data["observations"],
        "next_observations": data["next_observations"],
        "dones": data["dones"],
        "agent_state": agent_state,
        "recurrent": data["recurrent"],
        "rewards": data["rewards"],
        "gamma": data["gamma"],
        "alpha": jnp.exp(log_alpha),
    }


@pytest.fixture
def setup_policy_update_inputs(setup_loss_inputs):
    data = setup_loss_inputs

    agent_state = AgentState(
        rng=data["rng"],
        actor_state=data["actor_state"],
        critic_states=tuple(
            MaybeRecurrentTrainState(
                state=cs,
                hidden_state=chs,
            )
            for cs, chs in zip(data["critic_states"], data["target_critic_states"])
        ),
        target_critic_states=data["target_critic_states"],
        buffer_state=None,
        alpha=jnp.ones((1,)),
    )

    return {
        "observations": data["observations"],
        "done": data["dones"],
        "agent_state": agent_state,
        "recurrent": data["recurrent"],
    }


@pytest.fixture
def setup_alpha_update_inputs(setup_loss_inputs):
    data = setup_loss_inputs

    agent_state = AgentState(
        rng=data["rng"],
        actor_state=data["actor_state"],
        critic_states=tuple(
            MaybeRecurrentTrainState(
                state=cs,
                hidden_state=chs,
            )
            for cs, chs in zip(data["critic_states"], data["target_critic_states"])
        ),
        target_critic_states=data["target_critic_states"],
        buffer_state=None,
        alpha=data["alpha"],
    )

    target_entropy = -float(data["action_dim"])
    return {
        "agent_state": agent_state,
        "observations": data["observations"],
        "target_entropy": target_entropy,
        "recurrent": data["recurrent"],
        "done": data["dones"],
    }


def test_update_value_functions_runs(setup_update_inputs):
    updated_agent_state, aux = update_value_functions(**setup_update_inputs)

    assert isinstance(updated_agent_state, AgentState)
    assert updated_agent_state.critic_states is not None
    assert updated_agent_state.rng is not None
    assert jnp.isfinite(aux["critic_loss"])


def test_update_policy_runs(setup_policy_update_inputs):
    updated_agent_state, aux = update_policy(**setup_policy_update_inputs)

    assert isinstance(updated_agent_state, AgentState)
    assert updated_agent_state.actor_state is not None
    assert updated_agent_state.actor_state.state.params is not None
    assert updated_agent_state.rng is not None
    assert jnp.isfinite(aux["policy_loss"])


def test_update_alpha_runs(setup_alpha_update_inputs):
    updated_agent_state, aux = update_temperature(**setup_alpha_update_inputs)

    assert isinstance(updated_agent_state.alpha, TrainState)
    assert "alpha_loss" in aux
    assert "alpha" in aux
    assert "log_alpha" in aux
    assert jnp.isfinite(aux["alpha_loss"])
