import jax
import jax.numpy as jnp
from brax import envs

from jaxppo.wrappers import NormalizeVecObservationBrax, NormalizeVecRewardBrax


def test_normalize_rew_brax():
    env = envs.create("inverted_pendulum")
    normalized_rew_env = NormalizeVecRewardBrax(env, gamma=0.99)

    key = jax.random.PRNGKey(0)
    state = normalized_rew_env.reset(key)
    wrapped_rew, rew = state.reward, env.reset(key).reward

    action = 0.0
    new_state = normalized_rew_env.step(state, action)


def test_normalize_obs_brax():
    env = envs.create("inverted_pendulum")
    normalized_obs_env = NormalizeVecObservationBrax(env)

    key = jax.random.PRNGKey(0)
    state = normalized_obs_env.reset(key)
    wrapped_obs, obs = state.obs, env.reset(key).obs
    assert not (jnp.array_equal(wrapped_obs, obs))

    action = 0.0
    new_state = normalized_obs_env.step(state, action)
    assert not (jnp.array_equal(new_state.obs, env.step(state, action).obs))
