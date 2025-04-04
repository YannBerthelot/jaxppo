from typing import Any, Optional, TypeAlias

import gymnasium as gym
import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import wandb
from brax.generalized.base import State
from gymnax import EnvParams
from gymnax.environments.environment import Environment

from jaxppo.networks.networks import get_pi
from jaxppo.networks.networks_RNN import init_hidden_state
from jaxppo.utils import (
    check_env_is_brax,
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)

UpdateState: TypeAlias = Any


def get_tracking_camera():
    camera = mujoco._structs.MjvCamera()
    camera.type = mujoco._enums.mjtCamera.mjCAMERA_FIXED
    camera.trackbodyid = 0
    camera.fixedcamid = 0
    return camera


def get_all_non_none_keys(pipeline_state):
    keys = []
    for key in pipeline_state.__dataclass_fields__.keys():
        val = pipeline_state.__dict__[key]
        if val is not None:
            if isinstance(val, jnp.ndarray):
                keys.append(key)
            else:
                keys += get_all_non_none_keys(val)
    return keys


def check_attribute_exists(obj: Any, attr: str) -> bool:
    try:
        obj.__getattr__(attr)
    except AttributeError:
        return False
    else:
        return True


def save_video_to_wandb(
    env_id: Optional[str],
    env: Environment,
    recurrent: bool,
    update_state: UpdateState,
    rng: jax.Array,
    params: Optional[EnvParams] = None,
) -> None:
    """Generate an episode using the current agent state and log its video to wandb"""

    done = False

    continuous = check_if_environment_has_continuous_actions(env)
    if check_attribute_exists(env, "render"):
        env_render = env
        rng, reset_key = jax.random.split(rng)
        if check_env_is_gymnax(env_render):
            obs_render, state = env.reset(reset_key)
            frames = []
        else:
            state = env.reset(reset_key)
            obs_render, _, done, _ = (
                state.obs,
                state.reward,
                state.done,
                state.info,
            )

            non_None_keys = get_all_non_none_keys(state.pipeline_state)

    elif env_id is not None:
        env_render = gym.make(env_id, render_mode="rgb_array")
        obs_render, _ = env_render.reset(seed=42)
    else:
        raise ValueError(
            f"The environment {env} has no rendering, and the fallback env_id"
            " was not provided"
        )

    if env_id is not None:
        env_render = gym.make(env_id, render_mode="rgb_array")
        obs_render, _ = env_render.reset(seed=42)

    frames = []
    max_iteration = 1000

    def flatten_state(pipeline_state):
        vals = [
            (
                jnp.ravel(pipeline_state.__dict__[key])
                if isinstance(pipeline_state.__dict__[key], jnp.ndarray)
                else jnp.ravel(flatten_state(pipeline_state.__dict__[key]))
            )
            for key in pipeline_state.__dataclass_fields__.keys()
            if key in non_None_keys
        ]
        return jnp.concatenate(vals)

    def get_flattened_shape_of_state(pipeline_state: State) -> int:
        return flatten_state(pipeline_state).shape[0]

    def restore_state(pipeline_state, flattened_state):
        # Initialize an index pointer to keep track of where we are in the flattened array
        idx = 0
        # Dictionary to hold all the fields and their values
        restored_dict = {}

        # Get the class of the pipeline_state
        pipeline_state_class = pipeline_state.__class__

        # Helper function to recursively restore each field
        def restore_field(key, flattened_state, restored_dict):
            nonlocal idx
            field = pipeline_state_class.__dataclass_fields__[key]

            if key in non_None_keys:
                if field.type == jnp.ndarray:
                    # For fields that are arrays, slice out the corresponding part of flattened_state
                    example_value = getattr(
                        pipeline_state, key
                    )  # Get the example value from the pipeline_state instance
                    shape = example_value.shape  # Get the shape from the example value
                    restored_dict[key] = flattened_state[
                        idx : idx + np.prod(shape)
                    ].reshape(shape)
                    idx += np.prod(shape)
                else:
                    # For nested dataclass fields, call restore_state recursively
                    restored_dict[key] = restore_state(
                        getattr(pipeline_state, key),
                        flattened_state[idx:],
                    )
                    # Update idx by the size of the nested structure
                    idx += sum(
                        np.prod(f.shape)
                        for f in restored_dict[key].__dataclass_fields__.values()
                    )
            else:
                # For keys that had None values, we need to set them back to None
                restored_dict[key] = None
            return restored_dict

        # Iterate over all the fields and restore them

        for key in pipeline_state_class.__dataclass_fields__.keys():
            restored_dict = restore_field(key, flattened_state, restored_dict)

        # Instantiate the object with the restored dictionary
        # This assumes that the restored_dict contains all the required arguments
        return pipeline_state_class(**restored_dict)

    if check_env_is_brax(env_render):
        trajectory = (
            jnp.zeros(
                (max_iteration, get_flattened_shape_of_state(state.pipeline_state))
            )
            * jnp.nan
        )
    else:
        trajectory = None

    actor_hidden_state = (
        init_hidden_state(
            update_state.actor_hidden_state.shape[-1], num_envs=1, rng=rng
        )
        if recurrent
        else None
    )
    i = 0

    # Check if the environment is Gymnax or Brax
    if check_env_is_gymnax(env_render) or check_env_is_brax(env_render):
        # JAX-based loop for Gymnax and Brax
        def body_fn(carry):
            i, done, state, obs_render, rng, actor_hidden_state, trajectory, frames = (
                carry
            )
            # Perform the actions inside the loop

            rng, action_key = jax.random.split(rng)
            pi, actor_hidden_state = get_pi(
                update_state.actor_state,
                update_state.actor_state.params,
                obs_render[jnp.newaxis, :] if recurrent else obs_render,
                actor_hidden_state if recurrent else None,
                done_jax.reshape(-1, 1) if recurrent else None,
                recurrent,
            )

            action = pi.sample(seed=action_key)

            if check_env_is_gymnax(env_render):
                obs_render, state, _, done, _ = env_render.step(
                    rng, state, action, params
                )
                frames, screen, _ = env_render.render(
                    screen, state, params, frames, None
                )
            elif check_env_is_brax(env_render):  # brax
                state = env_render.step(state, action)
                obs_render, _, done, _ = (
                    state.obs,
                    state.reward,
                    state.done,
                    state.info,
                )

                flattened_state = flatten_state(state.pipeline_state)
                trajectory = trajectory.at[i].set(flattened_state)
                done = jnp.array(done) if isinstance(done, bool) else done
            return (
                i + 1,
                done,
                state,
                obs_render,
                rng,
                actor_hidden_state,
                trajectory,
                frames,
            )

        def cond_fn(carry):
            done = carry[1]
            return jnp.logical_not(done)

        # Using jax.lax.while_loop
        _, done, _, _, _, _, trajectory, frames = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (i, done, state, obs_render, rng, actor_hidden_state, trajectory, frames),
        )

        if check_env_is_brax(env_render):
            trajectory = [
                restore_state(state.pipeline_state, frame) for frame in trajectory
            ]

            frames = env_render.render(
                trajectory=trajectory, camera=get_tracking_camera()
            )

    else:
        done = False
        # Regular Python-based loop for other environments
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
            action = action if continuous else action.item()
            obs_render, _, terminated, truncated, _ = env_render.step(action)
            done = terminated | truncated
            new_frames = env_render.render()
            frames.append(new_frames)

    frames = np.array(frames).swapaxes(1, 3).swapaxes(2, 3)

    # Assume `frames` is a numpy array of shape (1000, 3, 240, 320)
    # Example: frames = np.random.randint(0, 256, (1000, 3, 240, 320), dtype=np.uint8)
    # Prepare output file name
    output_filename = (  # You can change the format to .avi or .mp4, etc.
        "output_video.mp4"
    )

    # Set the frames per second (FPS)
    fps = 30  # You can adjust this as needed

    # Initialize the video writer
    with imageio.get_writer(output_filename, fps=fps) as writer:
        # Loop through each frame and write to the video
        for i in range(frames.shape[0]):
            frame = frames[i]

            # Reorder the dimensions to (height, width, channels)
            frame_rgb = np.transpose(
                frame, (1, 2, 0)
            )  # Change shape from (3, 240, 320) to (240, 320, 3)

            # Ensure the frame is in uint8 format (values between 0 and 255)
            frame_rgb = frame_rgb.astype(np.uint8)

            # Write the frame to the video
            writer.append_data(frame_rgb)

    wandb.log({"video": wandb.Video(data_or_path=output_filename, format="mp4")})
