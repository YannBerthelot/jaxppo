import pdb
from typing import Tuple

import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class Buffer:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


def init_buffer(
    num_steps: int,
    num_envs: int,
    observation_space_shape: Tuple[int],
    action_space_shape: Tuple[int],
) -> Buffer:
    """Init the buffer with zero arrays of the right dimension for all values to track"""
    return Buffer(
        obs=jnp.zeros((num_steps, num_envs) + observation_space_shape),
        actions=jnp.zeros((num_steps, num_envs) + action_space_shape),
        logprobs=jnp.zeros((num_steps, num_envs)),
        dones=jnp.zeros((num_steps, num_envs)),
        values=jnp.zeros((num_steps, num_envs)),
        advantages=jnp.zeros((num_steps, num_envs)),
        returns=jnp.zeros((num_steps, num_envs)),
        rewards=jnp.zeros((num_steps, num_envs)),
    )


def insert_buffer(buffer: Buffer, step: int, feature_dict: dict) -> Buffer:
    """
    Insert new values into the buffer at the given step.
    Values are to be fed using a dict following this template:
        {value_name: new_value}
        e.g.
        {"obs":[7.3,1.5], "dones":[0.,1.]}
    """
    replace_dict = {
        key: buffer.__dict__[key].at[step].set(value)
        for key, value in feature_dict.items()
    }
    return buffer.replace(**replace_dict)


def update_gae_advantages(
    buffer: Buffer, next_done: int, next_value: float, gamma: float, gae_lambda: float
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
        buffer = buffer.replace(advantages=buffer.advantages.at[t].set(lastgaelam))
    return buffer


def update_returns(buffer: Buffer) -> Buffer:
    """Save returns as advantages + values"""
    return buffer.replace(returns=buffer.advantages + buffer.values)
