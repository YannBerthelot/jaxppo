"""Sanity checks on the behavior of gymnax"""
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
from jax import random


def test_gymnax_and_gymnasium_match():
    """check that gymnax and gymnasium env work the same"""
    env_id = "CartPole-v1"
    seed = 42
    key = random.PRNGKey(seed)
    env_gx, env_params = gymnax.make(env_id)
    env_gym = gymnasium.make(env_id)

    obs_gym, _ = env_gym.reset(seed=seed)
    obs_gx, env_state = env_gx.reset(key, env_params)
    env_state = env_state.__class__(
        *jnp.array(obs_gym, dtype=jnp.float32), time=jnp.array(0)
    )
    obs_gx = jnp.array(obs_gym, dtype=jnp.float32)
    assert jnp.allclose(obs_gym, obs_gx)
    action = 0
    done = False
    t = 0
    while not done:
        obs_gx, env_state, reward_gx, done_gx, _ = env_gx.step(
            key, env_state, action, env_params
        )
        obs_gym, reward_gym, terminated, truncated, _ = env_gym.step(action)
        done_gym = terminated | truncated
        done = done_gym
        if not done:
            assert jnp.array_equal(done_gx, done_gym, done_gx)
            assert jnp.array_equal(reward_gx, reward_gym)
            assert jnp.allclose(obs_gym, obs_gx)

        t += 1
