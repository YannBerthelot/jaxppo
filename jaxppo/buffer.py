"""Buffer definition and methods"""
from typing import Tuple

import jax.numpy as jnp
from flax import struct
from jax import Array, jit


class Buffer(struct.PyTreeNode):  # type : ignore
    """Store values as we go during rollout"""

    obs: Array
    actions: Array
    logprobs: Array
    dones: Array
    values: Array
    advantages: Array
    returns: Array
    rewards: Array


def init_buffer(
    num_steps: int,
    num_envs: int,
    observation_space_shape: Tuple[int],
    action_space_shape: Tuple[int],
) -> Buffer:
    """Init the buffer with zero arrays of the right dimension for all values to \
        track"""
    return Buffer(
        obs=jnp.zeros((num_steps, num_envs) + observation_space_shape),
        actions=jnp.zeros((num_steps, num_envs) + action_space_shape, dtype=jnp.int32),
        logprobs=jnp.zeros((num_steps, num_envs)),
        dones=jnp.zeros((num_steps, num_envs)),
        values=jnp.zeros((num_steps, num_envs)),
        advantages=jnp.zeros((num_steps, num_envs)),
        returns=jnp.zeros((num_steps, num_envs)),
        rewards=jnp.zeros((num_steps, num_envs)),
    )


@jit
def insert_buffer(buffer: Buffer, step_idx: int, **kwargs) -> Buffer:
    """
    Insert new values into the buffer at the given step.
    Values are to be fed using a dict following this template:
        {value_name: new_value}
        e.g. for a single env
        {"obs":[7.3, 1.5], "dones":[0.,1.]}
        e.g. for 2 envs
        {"obs":[[6.2, 3.2], [4.5, 3.2]], "dones":[0., 1.]}

    """
    # for key, val in kwargs.items():
    #     if not isinstance(val, (float, int)):
    #         assert buffer.__dict__[key].shape[1] == val.shape[0], (
    #             f"{key} is problematic, buffer shape {buffer.__dict__[key].shape},"
    #             f" kwargs shape {val.shape}"
    #         )
    replace_dict = {
        key: buffer.__dict__[key].at[step_idx].set(value)
        for key, value in kwargs.items()
    }
    return buffer.replace(**replace_dict)


@jit
def update_gae_advantages(
    buffer: Buffer, next_done: int, next_value: Array, gamma: float, gae_lambda: float
) -> Buffer:
    """
    Updates GAE advantages in the buffer using :
    - the content of the buffer
    - the next done flag (0 or 1)
    - the value of the next_state.
    - gamma (0<=gamma<=1): the discount factor
    - gae_lambda (0<=gae_lambda<=1) : the gae hyperparameter between bias and variance
    """
    # Reset advantages values
    buffer = buffer.replace(advantages=buffer.advantages.at[:].set(0.0))
    # Compute advantage using generalized advantage estimate
    lastgaelam = 0
    num_steps = buffer.dones.shape[0]
    # gaes = []
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - buffer.dones[t + 1]
            nextvalues = buffer.values[t + 1]
        delta = (
            buffer.rewards[t] + gamma * nextvalues * nextnonterminal - buffer.values[t]
        )
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        # gaes.append(lastgaelam)
        buffer = buffer.replace(advantages=buffer.advantages.at[t].set(lastgaelam))
    return buffer


@jit
def update_returns(buffer: Buffer) -> Buffer:
    """Save returns as advantages + values"""
    return buffer.replace(returns=buffer.advantages + buffer.values)
