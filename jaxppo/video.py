from typing import Any, Optional, TypeAlias

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from gymnax import EnvParams
from gymnax.environments.environment import Environment

from jaxppo.networks.networks import get_pi
from jaxppo.networks.networks_RNN import init_hidden_state

UpdateState: TypeAlias = Any


def save_video_to_wandb(
    env_id: Optional[str],
    env: Environment,
    recurrent: bool,
    update_state: UpdateState,
    rng: jax.Array,
    params: EnvParams,
) -> None:
    """Generate an episode using the current agent state and log its video to wandb"""

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
    actor_hidden_state = (
        init_hidden_state(
            update_state.actor_hidden_state.shape[-1], num_envs=1, rng=rng
        )
        if recurrent
        else None
    )

    while not done:
        done_jax = jnp.array(done) if isinstance(done, bool) else done
        rng, action_key = jax.random.split(rng)
        pi, actor_hidden_state = get_pi(
            update_state.actor_state,
            update_state.actor_state.params,
            obs_render[jnp.newaxis, :][jnp.newaxis, :] if recurrent else obs_render,
            actor_hidden_state if recurrent else None,
            done_jax.reshape(-1, 1) if recurrent else None,
            recurrent,
        )

        action = pi.sample(seed=action_key)
        if isinstance(env_id, str):
            obs_render, _, terminated, truncated, _ = env_render.step(action)
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
