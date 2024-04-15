"""Helpers for weights&biases logging"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp

import wandb

os.environ["WANDB_SILENT"] = "false"
os.environ["WANDB_START_METHOD"] = "fork"


@dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    project_name: str
    run_name: str
    config: dict
    mode: str = "online"
    group_name: Optional[str] = None


def init_logging(logging_config: LoggingConfig, folder: str = "wandb"):
    """Init the wandb run with the logging config"""

    wandb.init(
        project=logging_config.project_name,
        group=logging_config.group_name,
        name=logging_config.run_name,
        save_code=False,
        monitor_gym=False,
        config=logging_config.config,
        mode=logging_config.mode,
        dir=folder,
    )


def wandb_log(info, metrics, num_envs):
    """Extract meaningful values from the info buffer and log them into wandb"""
    return_values = info["returned_episode_returns"][
        info["returned_episode"]
    ]  # get episodes that have finished
    timestep = jnp.max(info["timestep"]) * num_envs
    wandb.log(
        {"Train/mean_returns_over_batch": jnp.mean(return_values), "timestep": timestep}
    )
    (
        actor_loss,
        critic_loss,
        simple_actor_loss,
        entropy_loss,
        clip_fraction,
        advantages,
        targets,
        values,
    ) = metrics
    losses_dict = {
        "Train/advantages": advantages.mean(),
        "Train/clip_fraction": clip_fraction.mean(),
        "Train/values": values.mean(),
        "Train/targets": targets.mean(),
        "Losses/actor_loss": actor_loss.mean(),
        "Losses/critic_loss": critic_loss.mean(),
        "Losses/actor loss without entropy": simple_actor_loss.mean(),
        "Losses/entropy loss": entropy_loss.mean(),
    }
    jax.debug.callback(log_variables, losses_dict)


def log_model(model_path: Any, name: str):
    """Log the model weights to wandb"""
    artifact = wandb.Artifact(name=name, type="model")
    artifact.add_file(local_path=model_path)  # Add dataset directory to artifact
    wandb.log_artifact(artifact)  # Logs the artifact version "my_data:v0"


def log_variables(variables_to_log: dict, commit: bool = True):
    """Log variables (in form of a dict of names and values) into wandb.
    Commit to finish a step."""
    wandb.log(variables_to_log, commit=commit)


def wandb_test_log(rewards: jax.Array):
    """Log the test results into wandb"""
    hist = wandb.Histogram(rewards.tolist())
    wandb.log({"Test/Episodic reward": hist})
    wandb.run.summary["mean_test_reward"] = jnp.mean(rewards)


def finish_logging():
    """Terminate the wandb run to start a new one"""
    wandb.finish()
