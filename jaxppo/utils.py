"""Helper functions for various modules"""
from functools import partial
from typing import Any, Callable, Tuple, cast

import gymnasium as gym
import gymnax
import jax
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnax import EnvParams
from gymnax.environments.environment import Environment
from jax import random

from jaxppo.wrappers import LogWrapper


def get_env_action_shape(
    env: gym.Env | SyncVectorEnv | Environment | LogWrapper,
) -> tuple[int] | tuple:
    """
    Get the shape of the action space of a gym env
    (Number of actions if discrete, else the shape of the continuous actions)
    """
    if isinstance(env, (Environment, LogWrapper)):
        action_space = env.action_space()
    else:
        action_space = (
            env.single_action_space if "is_vector_env" in dir(env) else env.action_space  # type: ignore[union-attr]
        )
    if isinstance(action_space, (gymnax.environments.spaces.Discrete)):
        action_shape = ()
    elif isinstance(action_space, gym.spaces.Discrete):
        action_shape = (1,)  # type: ignore[assignment]
    else:
        action_shape = cast(Tuple[int], action_space.shape)  # type: ignore[assignment]
    return action_shape


def get_env_observation_shape(
    env: gym.Env | SyncVectorEnv | Environment | LogWrapper, env_params=None
) -> Tuple[int]:
    """
    Get the shape of the observation space of a gym env (or Vec Env)
    """
    if isinstance(env, (Environment, LogWrapper)):
        if env_params is None:
            raise ValueError("No env params provided for gymnax env")
        observation_space = env.observation_space(env_params)

    else:
        observation_space = (
            env.single_observation_space
            if "is_vector_env" in dir(env)
            else env.observation_space  # type: ignore[union-attr]
        )
    return cast(Tuple[int], observation_space.shape)


def sample_obs_space(
    env: Environment | gym.Env | SyncVectorEnv | LogWrapper,
    env_params: EnvParams = None,
) -> Any:
    """Sample an action from an environment"""
    if isinstance(env, (Environment, LogWrapper)):
        key = random.PRNGKey(42)
        return env.observation_space(env_params).sample(key)
    else:
        observation_space = (
            env.single_observation_space
            if "is_vector_env" in dir(env)
            else env.observation_space  # type: ignore[union-attr]
        )
        return observation_space.sample()


def get_num_actions(
    env: gym.Env | SyncVectorEnv | Environment | LogWrapper,
) -> int:
    """Get the number of actions (discrete or continuous) in a gym env"""
    if isinstance(env, (Environment, LogWrapper)):
        action_space = env.action_space()
    else:
        action_space = (
            env.single_action_space
            if "is_vector_env" in dir(env)
            else env.action_space  # type: ignore[union-attr]
        )
    if isinstance(
        action_space, (gym.spaces.Discrete, gymnax.environments.spaces.Discrete)
    ):
        num_actions = int(action_space.n)
    else:
        action_shape = cast(
            Tuple[int], action_space.shape
        )  # guaranteed to have a tuple now
        num_actions = int(action_shape[0])
    return num_actions


def linear_schedule(
    count: int,
    initial_learning_rate: float,
    decay: float,
) -> float:
    """Returns the updated learning rate given the initial one, the decay and the \
        count of timesteps since start"""
    return initial_learning_rate - (decay * count)


def annealed_linear_schedule(
    count: int,
    initial_learning_rate: float,
    num_minibatches: int,
    update_epochs: int,
    num_updates: int,
) -> float:
    """Compute the anneal learning rate for the given count of elapsed steps and\
          hyperparameters"""
    # anneal learning rate linearly after one training iteration which contains
    # (args.num_minibatches * args.update_epochs) gradient updates
    frac = 1.0 - (count // (num_minibatches * update_epochs)) / num_updates
    return initial_learning_rate * frac


def get_parameterized_schedule(
    linear_scheduler: Callable[..., float], **scheduler_kwargs: Any
) -> Callable[[int], float]:
    """Generates a schedule fit to be given to optax optimizers by pre-setting \
        hyperparameters. It will then only need the step count as input, \
        all other hyperparameters being already set."""
    return partial(linear_scheduler, **scheduler_kwargs)


def make_gymnax_env(
    env_id: str, seed: int
) -> tuple[
    Environment,
    EnvParams,
    tuple[random.PRNGKeyArray, random.PRNGKeyArray, random.PRNGKeyArray],
]:
    """Create a gymnax env and associated values for the given env id and seed"""
    rng = jax.random.PRNGKey(seed)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env, env_params = gymnax.make(env_id)
    return env, env_params, (key_reset, key_policy, key_step)
