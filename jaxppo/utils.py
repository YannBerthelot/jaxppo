"""Helper functions for various modules"""

import os
import pickle
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypeAlias, cast

import gymnasium as gym
import gymnax
import jax
import jax.numpy as jnp
from flax.serialization import from_state_dict, to_state_dict
from flax.training.train_state import TrainState
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnax import EnvParams
from gymnax.environments.environment import Environment as GymnaxEnvironment
from jax import random

from jaxppo.wandb_logging import log_model
from jaxppo.wrappers import BraxWrapper, GymnaxWrapper, LogWrapper, get_wrappers

try:
    from brax import envs as brax_envs

    BraxEnvironment = brax_envs.Env | brax_envs.wrappers.training.AutoResetWrapper
    Environment: TypeAlias = GymnaxEnvironment | BraxEnvironment
except ImportError:
    brax_envs = None
    Environment: TypeAlias = GymnaxEnvironment


def get_env_action_shape(
    env: gym.Env | SyncVectorEnv | Environment | LogWrapper,
) -> tuple[int] | tuple:
    """
    Get the shape of the action space of a gym env
    (Number of actions if discrete, else the shape of the continuous actions)
    """
    action_space = env.action_space()
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
    if env_params is None:
        raise ValueError("No env params provided for gymnax env")
    observation_space = env.observation_space(env_params)
    return cast(Tuple[int], observation_space.shape)


def sample_obs_space(
    env: Environment | gym.Env | SyncVectorEnv | LogWrapper,
    env_params: EnvParams = None,
) -> Any:
    """Sample an action from an environment"""
    key = random.PRNGKey(42)
    if isinstance(env, GymnaxEnvironment):
        return env.observation_space(env_params).sample(key)
    elif brax_envs is not None:
        if isinstance(env, BraxEnvironment):
            return env.reset(key)
    else:
        raise ValueError(f"Unsupported environment type {type(env)}")


def check_env_is_brax(env):
    return isinstance(env, BraxEnvironment) or issubclass(env.__class__, BraxWrapper)


def check_env_is_gymnax(env):
    return isinstance(env, GymnaxEnvironment) or issubclass(
        env.__class__, GymnaxWrapper
    )


def get_observation_space_shape(
    env: gym.Env | SyncVectorEnv | Environment | LogWrapper,
    env_params: Optional[EnvParams],
) -> int:
    if brax_envs is not None:
        if check_env_is_brax(env):
            return env.observation_size
    observation_space = env.observation_space(env_params)
    if isinstance(
        observation_space, (gym.spaces.Discrete, gymnax.environments.spaces.Discrete)
    ):
        observation_space_shape = int(observation_space.n)
    elif isinstance(
        observation_space, (gym.spaces.Box, gymnax.environments.spaces.Box)
    ):
        observation_space_shape = (
            int(observation_space.shape[0]) if len(observation_space.shape) > 0 else 1
        )
    else:
        action_shape = cast(
            Tuple[int], observation_space.shape
        )  # guaranteed to have a tuple now
        observation_space_shape = int(action_shape[0])
    return observation_space_shape


def get_num_actions(
    env: gym.Env | SyncVectorEnv | Environment | LogWrapper,
    params: Optional[EnvParams] = None,
) -> int:
    """Get the number of actions (discrete or continuous) in a gym env"""

    # TODO : add continuous
    if brax_envs is not None:
        if check_env_is_brax(env):
            return env.action_size
    action_space = env.action_space(params)
    if isinstance(
        action_space, (gym.spaces.Discrete, gymnax.environments.spaces.Discrete)
    ):
        num_actions = int(action_space.n)
    elif isinstance(action_space, (gym.spaces.Box, gymnax.environments.spaces.Box)):
        num_actions = int(action_space.shape[0]) if len(action_space.shape) > 0 else 1
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
) -> tuple[Environment, EnvParams, tuple[jax.Array, jax.Array, jax.Array],]:
    """Create a gymnax env and associated values for the given env id and seed"""
    rng = jax.random.PRNGKey(seed)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env, env_params = gymnax.make(env_id)
    return env, env_params, (key_reset, key_policy, key_step)


def save_model(
    actor: TrainState,
    critic: TrainState,
    idx: int,
    log: bool = True,
    save_folder: str = "./models",
):
    """Save the actor and critic state to the specified folder. Log the model to wandb if selected."""
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"update_{idx}.pkl")
    dict_to_save = {"actor": to_state_dict(actor), "critic": to_state_dict(critic)}
    with open(path, "wb") as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if log:
        log_model(path, "actor_critic_model")


def load_model(
    path: str, actor: TrainState, critic: TrainState
) -> tuple[TrainState, TrainState]:
    """Load actor and critic state from the save file in path"""
    with open(path, "rb") as handle:
        actor_and_critic = pickle.load(handle)
    return from_state_dict(actor, actor_and_critic["actor"]), from_state_dict(
        critic, actor_and_critic["critic"]
    )


def check_update_frequency(
    num_update: int, num_total_updates: int, frequency: Optional[int] = None
) -> bool:
    """Check wether or not to save according to the number of update.
    It will be true when either the number of update is a multiple of save_frequency \\
        or total number of updates has been reached"""
    if frequency is not None:
        cond = jnp.logical_or(
            num_update == num_total_updates, num_update % (frequency - 1) == 0
        )
    else:
        cond = num_update == num_total_updates
    return cond


env_dict = {
    "HalfCheetah-v4": "halfcheetah",
    "Hopper-v4": "hopper",
    "Humanoid-v4": "humanoid",
    "HumanoidStandup-v4": "humanoidstandup",
    "InvertedDoublePendulum-v4": "inverted_double_pendulum",
    "InvertedPendulum-v4": "inverted_pendulum",
    "Pusher-v4": "pusher",
    "Reacher-v4": "reacher",
    "Swimmer-v4": "swimmer",
    "Walker2d-v4": "walker",
    "Ant-v4": "ant",
}


def build_env_from_id(env_id: str, **kwargs) -> tuple[Environment, Optional[EnvParams]]:
    if env_id in gymnax.registered_envs:
        env, env_params = gymnax.make(env_id)
        return env, env_params
    if brax_envs is not None:
        if env_id in list(env_dict.values()):
            return brax_envs.create(env_id, **kwargs), None
        else:
            raise ValueError(f"Environment {env_id} not found in gymnax or mjx")
    else:
        raise ValueError(
            f"The env id ({env_id}) provided is not recognized as a valid gymnax env,"
            " cannot check for Mujoco as mjx is not installed."
        )


def prepare_env(
    env_id,
    continuous,
    gamma,
    episode_length: Optional[int] = None,
    env_params: Optional[EnvParams] = None,
):
    if isinstance(env_id, str):
        env, env_params = build_env_from_id(
            env_id, episode_length=1000 if episode_length is None else episode_length
        )
    else:  # env is assumed to be provided already built
        env = env_id
        env_id = None  # To prepare video saving
    if continuous:
        mode = "gymnax" if isinstance(env, GymnaxEnvironment) else "brax"
        ClipAction, NormalizeVecObservation, NormalizeVecReward = get_wrappers(mode)
        # env = ClipAction(env, low=-1.0, high=1.0)
        env = NormalizeVecObservation(env)
        # env = NormalizeVecReward(env, gamma)
    return env, env_params, env_id
