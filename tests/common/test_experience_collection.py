from functools import partial

import flashbax as fbx
import jax
import jax.numpy as jnp
import pytest
from gymnax.environments.classic_control import CartPole

from jaxppo.environment.environment_interaction import reset_env, step_env
from jaxppo.networks.networks import (
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
)
from jaxppo.sac.train_sac_2 import (
    AlphaProperties,
    BufferProperties,
    CollectorState,
    EnvironmentProperties,
    MaybeRecurrentTrainState,
    NetworkProperties,
    OptimizerProperties,
    collect_experience,
    get_buffer,
    init_agent,
)

NUM_ENVS = 4
ROLLOUT_LENGTH = 8


@pytest.fixture
def dummy_alpha_args():
    return AlphaProperties(alpha_init=0.1, learning_rate=3e-4)


@pytest.fixture
def dummy_env_args():
    env = CartPole()
    return EnvironmentProperties(
        env=env,
        env_params=env.default_params,
        num_envs=NUM_ENVS,
        continuous=False,
    )


@pytest.fixture
def dummy_network_args():
    return NetworkProperties(
        actor_architecture=("32", "32"),
        critic_architecture=("32", "32"),
        lstm_hidden_size=None,  # non-recurrent
    )


@pytest.fixture
def dummy_optimizer_args():
    return OptimizerProperties(
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )


@pytest.fixture
def dummy_buffer():
    dummy_buffer_props = BufferProperties(buffer_size=1000)
    return get_buffer(dummy_buffer_props.buffer_size, dummy_buffer_props.minibatch_size)


def test_collect_experience_scan(
    dummy_env_args,
    dummy_network_args,
    dummy_optimizer_args,
    dummy_buffer,
    dummy_alpha_args,
):
    key = jax.random.PRNGKey(42)

    (
        rng,
        actor_state,
        critic_state,
        target_critic_state,
        buffer_state,
        alpha_state,
    ) = init_agent(
        key=key,
        env_args=dummy_env_args,
        optimizer_args=dummy_optimizer_args,
        network_args=dummy_network_args,
        buffer=dummy_buffer,
        alpha_args=dummy_alpha_args,
    )

    mode = "gymnax"
    last_done = jnp.zeros(dummy_env_args.num_envs)
    reset_keys = (
        jax.random.split(rng, dummy_env_args.num_envs) if mode == "gymnax" else rng
    )
    last_obs, env_state = reset_env(
        reset_keys, dummy_env_args.env, mode, dummy_env_args.env_params
    )

    collector_state = CollectorState(
        rng=key,
        env_state=env_state,
        last_obs=last_obs,
        buffer_state=buffer_state,
        actor_state=actor_state,
        timestep=0,
        last_done=last_done,
    )

    scan_fn = partial(
        collect_experience,
        recurrent=False,
        mode="gymnax",
        env_args=dummy_env_args,
        buffer=dummy_buffer,
    )

    final_state, _ = jax.lax.scan(
        scan_fn, collector_state, xs=None, length=ROLLOUT_LENGTH
    )

    # Check final output validity
    assert isinstance(final_state, CollectorState)
    assert final_state.last_obs.shape[0] == NUM_ENVS
    assert final_state.timestep == ROLLOUT_LENGTH
    assert dummy_buffer.sample(final_state.buffer_state, rng).experience.first[
        "obs"
    ].shape == (
        1,
        NUM_ENVS,
        *final_state.last_obs.shape[1:],
    )
