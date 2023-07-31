"""Config for PPO agent"""
from typing import Any, Optional, Sequence

from gymnax.environments.environment import Environment
from gymnax.wrappers.purerl import GymnaxWrapper
from pydantic import BaseModel, ConfigDict, validator

from jaxppo.wandb_logging import LoggingConfig


class PPOConfig(BaseModel):
    """Config for PPO agent"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_timesteps: int
    num_steps: int
    num_envs: int
    env_id: str | Environment | GymnaxWrapper
    learning_rate: float
    num_minibatches: int = 4
    update_epochs: int = 4
    actor_architecture: Sequence[str] = ["64", "tanh", "64", "tanh"]
    critic_architecture: Sequence[str] = ["64", "relu", "relu", "tanh"]
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    logging_config: Optional[LoggingConfig] = None
    num_episode_test: int = 20
    env_params: Optional[Any] = None

    @validator("env_id")
    @classmethod
    def check_env_id(cls, value):
        """Check that value environment is a valid gymnax env id, gymnax env, or gymnax wrapped env"""
        if isinstance(value, str):
            return value

        elif isinstance(value, Environment) or issubclass(
            value.__class__, GymnaxWrapper
        ):
            return value
        else:
            raise ValueError(
                "Environment should be either a valid env id or a gymnax environment or"
                " a gymnax wrapped_env"
            )

    @validator("env_params")
    @classmethod
    def check_env_params(cls, value, values):
        """Check that the env params are of the proper type"""
        if value is None:
            if "env_id" in values:
                if not isinstance(values["env_id"], str):
                    raise ValueError("Missing EnvParams for pre-defined env")
            return value
        if "EnvParams" not in str(value.__class__):
            raise ValueError(
                f"env_params should be of a EnvParams type, got {type(value)}"
            )
        return value
