"""Type hints for the rnn agents"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

HiddenState = jax.Array


class Transition(NamedTuple):
    """The current rollout-buffer (storing transition from the environment rollouts)"""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


RunnerState = tuple[
    jnp.ndarray,  # actor_state
    jnp.ndarray,  # critic_state
    jnp.ndarray,  # env_state
    jnp.ndarray,  # last_obs
    jnp.ndarray,  # last_done
    jnp.ndarray,  # actor_hidden_state
    jnp.ndarray,  # critic_hidden_state
    jax.Array,  # rng
]

UpdateState = tuple[
    jnp.ndarray,  # actor_state
    jnp.ndarray,  # critic_state
    Transition,  # traj_batch
    jnp.ndarray,  # advantages
    jnp.ndarray,  # targets
    jax.Array,  # rng
    jnp.ndarray,  # actor_hidden_state
    jnp.ndarray,  # critic_hidden_state
]

BatchInfo = tuple[
    Transition, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]  # traj_batch, advantages, targets, actor_hidden_state, critic_hidden_state
