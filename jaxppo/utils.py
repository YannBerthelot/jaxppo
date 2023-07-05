"""Helper functions for various modules"""
from functools import partial
from typing import Any, Callable, Union

import gymnasium as gym


def get_env_action_shape(env: gym.Env) -> Union[int, tuple]:
    """
    Get the shape of the action space of a gym env
    (Number of actions if discrete, else the shape of the continuous actions)
    """
    return (
        (env.action_space.n,)
        if isinstance(env.action_space, gym.spaces.Discrete)
        else env.action_space.shape
    )


def get_num_actions(env: gym.Env) -> int:
    """Get the number of actions (discrete or continuous) in a gym env"""
    return (
        env.action_space.n
        if isinstance(env.action_space, gym.spaces.Discrete)
        else env.action_space.shape[0]
    )


def linear_schedule(initial_learning_rate: float, decay: float, count: int) -> float:
    """Returns the updated learning rate given the initial one, the decay and the \
        count of timesteps since start"""
    frac = 1 - (decay * count) / initial_learning_rate
    return initial_learning_rate * frac


def get_parameterized_schedule(
    linear_scheduler: Callable[[Any], float], **scheduler_kwargs: float
) -> Callable[[int], float]:
    """Generates a schedule fit to be given to optax optimizers by pre-setting \
        hyperparameters. It will then only need the step count as input, \
        all other hyperparameters being already set."""
    return partial(linear_scheduler, **scheduler_kwargs)
