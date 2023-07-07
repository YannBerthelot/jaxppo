"""Helpers for weights&biases logging"""
from dataclasses import dataclass

import wandb


@dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    project_name: str
    run_name: str
    config: dict


def init_logging(logging_config: LoggingConfig):
    """Init the wandb run with the logging config"""
    wandb.init(
        project=logging_config.project_name,
        name=logging_config.run_name,
        save_code=True,
        monitor_gym=True,
        config=logging_config.config,
    )


def log_variables(variables_to_log: dict, commit: bool = False):
    """Log variables (in form of a dict of names and values) into wandb.
    Commit to finish a step."""
    wandb.log(variables_to_log, commit=commit)
