"""Helper functions for various modules"""

import os
import pickle
from functools import partial
from typing import Any, Callable, Tuple, cast

import gymnasium as gym
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_state_dict, to_state_dict
from flax.training.train_state import TrainState
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnax import EnvParams
from gymnax.environments.environment import Environment
from jax import random

import wandb
from jaxppo.wandb_logging import log_model
from jaxppo.wrappers import LogWrapper


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
    return env.observation_space(env_params).sample(key)


def get_num_actions(
    env: gym.Env | SyncVectorEnv | Environment | LogWrapper,
) -> int:
    """Get the number of actions (discrete or continuous) in a gym env"""
    action_space = env.action_space()
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


def make_gymnax_env(env_id: str, seed: int) -> tuple[
    Environment,
    EnvParams,
    tuple[random.PRNGKeyArray, random.PRNGKeyArray, random.PRNGKeyArray],
]:
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


def check_update_frequency_for_video(num_update, num_updates, video_log_frequency):
    if video_log_frequency is not None:
        cond = jnp.logical_or(
            num_update == num_updates, num_update % (video_log_frequency - 1) == 0
        )
    else:
        cond = num_update == num_updates
    return cond


def check_update_frequency_for_saving(num_update, num_updates, save_frequency):
    if save_frequency is not None:
        cond = jnp.logical_or(
            num_update == num_updates, num_update % (save_frequency - 1) == 0
        )
    else:
        cond = num_update == num_updates
    return cond


def save_video_to_wandb(
    env_id,
    env,
    update_state,
    rng: jax.random.PRNGKeyArray,
    params,
):
    if isinstance(env_id, str):
        env_render = gym.make(env_id, render_mode="rgb_array")
        obs_render, _ = env_render.reset(seed=42)
    else:
        env_render = env
        rng, reset_key = jax.random.split(rng)
        obs_render, state = env.reset(reset_key)
        screen = None
        clock = None

    done = False
    frames = []
    FPS = 50

    while not done:
        rng, action_key = jax.random.split(rng)
        action = update_state.actor_state.apply_fn(
            update_state.actor_state.params, jnp.array(obs_render)
        ).sample(seed=action_key)

        if isinstance(env_id, str):
            obs_render, _, terminated, truncated, _ = env_render.step(action.item())
            done = terminated | truncated
            new_frames = env_render.render()
            frames.append(new_frames)
        else:
            obs_render, state, _, done, _ = env_render.step(rng, state, action, params)
            frames, screen, clock = env_render.render(
                screen, state, params, frames, clock
            )

    frames_correct_order = np.array(frames).swapaxes(1, 3).swapaxes(2, 3)
    wandb.log({"video": wandb.Video(frames_correct_order, fps=FPS)})
