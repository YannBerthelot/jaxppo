from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnames=["mode", "env", "env_params"])
def reset_env(
    rng: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> tuple[jax.Array, EnvState]:
    if mode == "gymnax":
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(rng, env_params)
    else:
        env_state = env.reset(rng)  # ✅ no vmap
        obsv = env_state.obs
    return obsv, env_state


@partial(
    jax.jit, static_argnames=["mode", "env", "env_params"], donate_argnames=["state"]
)
def step_env(
    rng: jax.Array,
    state: jax.Array,
    action: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Any]:
    if mode == "gymnax":
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng, state, action, env_params)
        done = jnp.float_(done)
    else:  # ✅ no vmap for brax
        env_state = env.step(state, action)
        obsv, reward, done, info = (
            env_state.obs,
            env_state.reward,
            env_state.done,
            env_state.info,
        )
    return obsv, env_state, reward, done, info
