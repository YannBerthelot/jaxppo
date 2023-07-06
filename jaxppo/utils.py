"""Helper functions for various modules"""
from functools import partial
from typing import Any, Callable, Tuple, Union, cast

import gymnasium as gym
from gymnasium.vector.sync_vector_env import SyncVectorEnv


def get_env_action_shape(env: Union[gym.Env, SyncVectorEnv]) -> tuple:
    """
    Get the shape of the action space of a gym env
    (Number of actions if discrete, else the shape of the continuous actions)
    """

    action_space = (
        env.single_action_space if "is_vector_env" in dir(env) else env.action_space  # type: ignore[union-attr]
    )
    if isinstance(action_space, gym.spaces.Discrete):
        action_shape = (1,)
    else:
        action_shape = cast(
            Tuple[int], action_space.shape
        )  # guaranteed to have a tuple now
    return action_shape


def get_env_observation_shape(env: Union[gym.Env, SyncVectorEnv]) -> tuple:
    """
    Get the shape of the observation space of a gym env (or Vec Env)
    """
    observation_space = (
        env.single_observation_space  # type: ignore[union-attr]
        if "is_vector_env" in dir(env)
        else env.observation_space
    )
    return cast(Tuple[int], observation_space.shape)


def get_num_actions(env: Union[gym.Env, SyncVectorEnv]) -> int:
    """Get the number of actions (discrete or continuous) in a gym env"""
    action_space = (
        env.single_action_space if "is_vector_env" in dir(env) else env.action_space  # type: ignore[union-attr]
    )
    if isinstance(action_space, gym.spaces.Discrete):
        num_actions = int(action_space.n)
    else:
        action_shape = cast(
            Tuple[int], action_space.shape
        )  # guaranteed to have a tuple now
        num_actions = int(action_shape[0])
    return num_actions


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


def _make_single_env(env_id: str, idx: int = 0, capture_video: bool = False):
    """Create a single env for the given env_id, index and parameters"""

    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{env_id}")
        else:
            env = gym.make(env_id)
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "only discrete action space is supported atm"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_envs(env_id: str, capture_video: bool, num_envs: int):
    """Create a stack of num_envs environments (can also be used to create 1 env)"""
    return gym.vector.SyncVectorEnv(
        [_make_single_env(env_id, i, capture_video) for i in range(num_envs)]
    )
