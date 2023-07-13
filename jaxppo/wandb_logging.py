"""Helpers for weights&biases logging"""
from dataclasses import dataclass

import jax.numpy as jnp

import wandb


@dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    project_name: str
    run_name: str
    config: dict


def init_logging(logging_config: LoggingConfig):
    """Init the wandb run with the logging config"""
    wandb.init(  # type: ignore[attr-defined]
        project=logging_config.project_name,
        name=logging_config.run_name,
        save_code=True,
        monitor_gym=True,
        config=logging_config.config,
    )


def wandb_log(info):
    """Extract meaningful values from the info buffer and log them into wandb"""
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    timestep = jnp.max(info["timestep"]) * 8
    wandb.log(
        {"mean_returns_over_batch": jnp.mean(return_values), "timestep": timestep}
    )


def log_variables(variables_to_log: dict, commit: bool = False):
    """Log variables (in form of a dict of names and values) into wandb.
    Commit to finish a step."""
    wandb.log(variables_to_log, commit=commit)  # type: ignore[attr-defined]


def finish_logging():
    """Terminate the wandb run to start a new one"""
    wandb.finish()
