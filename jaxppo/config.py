"""Config for PPO agent"""
import json
from typing import Any, Optional, Sequence

from gymnax.environments.environment import Environment
from gymnax.wrappers.purerl import GymnaxWrapper
from pydantic import BaseModel, ConfigDict, field_validator

from jaxppo.wandb_logging import LoggingConfig


class PPOConfig(BaseModel):
    """Config for PPO agent"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_timesteps: int
    num_steps: int
    num_envs: int
    env_params: Optional[Any] = None
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

    @field_validator("env_id")
    @classmethod
    def check_env_id_and_params(cls, env_id, info):
        """Check that value environment is a valid gymnax env id, gymnax env, or gymnax wrapped env"""

        env_params = info.data["env_params"]
        if isinstance(env_id, str):
            return env_id
        elif isinstance(env_id, Environment) or issubclass(
            env_id.__class__, GymnaxWrapper
        ):
            if info.data is None:
                raise ValueError("Missing EnvParams for pre-defined env")
            if "EnvParams" not in str(env_params.__class__):
                raise ValueError(
                    f"env_params should be of a EnvParams type, got {type(env_params)}"
                )
            return env_id
        else:
            raise ValueError(
                "Environment should be either a valid env id or a gymnax"
                " environment or a gymnax wrapped_env"
            )

    def to_dict(self):
        """Converts the Pydantic config to a dict"""
        return json.loads(self.model_dump_json())
