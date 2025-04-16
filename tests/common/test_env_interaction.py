from functools import partial

import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from brax.envs.base import State as BraxState
from gymnax.environments.classic_control.cartpole import CartPole
from gymnax.environments.environment import EnvState

from jaxppo.environment.environment_interaction import (  # Update this to match your import
    reset_env,
    step_env,
)

NUM_ENVS = 4


# ---------------- Fixtures ---------------- #


@pytest.fixture
def gymnax_env_and_params():
    env = CartPole()
    params = env.default_params
    return env, params, "gymnax"


@pytest.fixture
def brax_env_and_params():
    env = create_brax_env("ant", batch_size=NUM_ENVS)  # MuJoCo-style env
    return env, None, "brax"


# ----------------- Shared Tests ----------------- #


def _test_reset(env, env_params, mode):
    rng = (
        jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
        if mode == "gymnax"
        else jax.random.PRNGKey(0)
    )

    obsv, env_state = reset_env(
        rng=rng,
        env=env,
        env_params=env_params,
        mode=mode,
    )

    assert isinstance(obsv, jax.Array)
    assert isinstance(env_state, EnvState if mode == "gymnax" else BraxState)
    assert obsv.shape[0] == NUM_ENVS


def _test_step(env, env_params, mode):
    rng = (
        jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
        if mode == "gymnax"
        else jax.random.PRNGKey(0)
    )

    obsv, env_state = reset_env(rng, env, mode, env_params)

    if mode == "gymnax":
        action = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)
    else:  # Brax envs use continuous actions
        action_dim = env.action_size
        action = jnp.zeros((NUM_ENVS, action_dim), dtype=jnp.float32)

    rng = (
        jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
        if mode == "gymnax"
        else jax.random.PRNGKey(0)
    )

    obsv2, env_state2, reward, done, info = step_env(
        rng=rng,
        state=env_state,
        action=action,
        env=env,
        mode=mode,
        env_params=env_params,
    )

    assert isinstance(reward, jax.Array)
    assert obsv2.shape[0] == NUM_ENVS
    assert reward.shape == (NUM_ENVS,)
    assert done.shape == (NUM_ENVS,)


# ----------------- Individual Tests ----------------- #


def test_reset_env_gymnax(gymnax_env_and_params):
    env, env_params, mode = gymnax_env_and_params
    _test_reset(env, env_params, mode)


def test_step_env_gymnax(gymnax_env_and_params):
    env, env_params, mode = gymnax_env_and_params
    _test_step(env, env_params, mode)


def test_reset_env_brax(brax_env_and_params):
    env, env_params, mode = brax_env_and_params
    _test_reset(env, env_params, mode)


def test_step_env_brax(brax_env_and_params):
    env, env_params, mode = brax_env_and_params
    _test_step(env, env_params, mode)


# ----------------- JIT and VMAP Compatibility ----------------- #


def test_jit_compatibility(gymnax_env_and_params):
    env, env_params, mode = gymnax_env_and_params
    rng = jax.random.split(jax.random.PRNGKey(123), NUM_ENVS)

    jitted_reset = jax.jit(reset_env, static_argnames=["mode", "env", "env_params"])
    obsv, state = jitted_reset(rng, env, mode, env_params)

    assert obsv.shape[0] == NUM_ENVS


def test_jit_brax(brax_env_and_params):
    env, env_params, mode = brax_env_and_params
    rng = jax.random.PRNGKey(1234)

    jitted_reset = jax.jit(reset_env, static_argnames=["mode", "env", "env_params"])
    obsv, state = jitted_reset(rng, env, mode, env_params)

    assert obsv.shape[0] == NUM_ENVS
