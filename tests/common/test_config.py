"""Test pydantic config"""

import gymnax

from jaxppo.ppo.ppo_config import PPOConfig


def test_serializing_env_id():
    """Check that it's possible to serialize the config"""
    env_id = "CartPole-v1"
    config = PPOConfig(
        total_timesteps=int(1e6),
        num_steps=8,
        num_envs=4,
        num_episode_test=10,
        env_id=env_id,
        learning_rate=1e-3,
    )
    expected_config = {
        "total_timesteps": int(1e6),
        "num_steps": 8,
        "num_envs": 4,
        "num_episode_test": 10,
        "learning_rate": 1e-3,
        "env_params": None,
        "env_id": "CartPole-v1",
    }

    assert expected_config.items() <= config.to_dict().items()


def test_serializing_env():
    """Check that it's possible to serialize the config with normally un-hashable types"""
    env, env_params = gymnax.make("CartPole-v1")
    config = PPOConfig(
        total_timesteps=int(1e6),
        num_steps=8,
        num_envs=4,
        num_episode_test=10,
        env_params=env_params,
        env_id=env,
        learning_rate=1e-3,
    )
    expected_config = {
        "total_timesteps": int(1e6),
        "num_envs": 4,
        "num_steps": 8,
        "num_episode_test": 10,
        "learning_rate": 1e-3,
        "env_params": str(env_params.__class__),
        "env_id": str(env.__class__),
    }
    assert expected_config.items() <= config.to_dict().items()
