"""Type hints for the rnn agents"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import random

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


# class RunnerState(NamedTuple):
#     """The current agent state used to step the env"""

#     actor_state: jnp.ndarray
#     critic_state: Optional[jnp.ndarray]
#     env_state: jnp.ndarray
#     last_obs: jnp.ndarray
#     last_done: jnp.ndarray
#     actor_hstate: jnp.ndarray
#     critic_hstate: jnp.ndarray
#     rng: random.PRNGKeyArray

RunnerState = tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    random.PRNGKeyArray,
]


# class UpdateState(NamedTuple):
#     """The current state of the updated parameters and the required variables\
#           for the update"""

#     actor_state: jnp.ndarray
#     critic_state: Optional[jnp.ndarray]
#     traj_batch: Transition
#     advantages: jnp.ndarray
#     targets: jnp.ndarray
#     rng: random.PRNGKeyArray
#     actor_hstate: jnp.ndarray
#     critic_hstate: jnp.ndarray


UpdateState = tuple[
    jnp.ndarray,
    Optional[jnp.ndarray],
    Transition,
    jnp.ndarray,
    jnp.ndarray,
    random.PRNGKeyArray,
    jnp.ndarray,
    jnp.ndarray,
]


# class BatchInfo(NamedTuple):
#     traj_batch: Transition
#     advantages: jnp.ndarray
#     targets: jnp.ndarray
#     actor_hstate: Optional[jnp.ndarray] = None
#     critic_hstate: Optional[jnp.ndarray] = None


BatchInfo = tuple[Transition, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
