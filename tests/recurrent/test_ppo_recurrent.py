"""Test ppo"""

import os

import gymnasium as gym
import gymnax
import jax.numpy as jnp
import pytest
import wandb
from gymnax.wrappers.purerl import FlattenObservationWrapper  # pylint: disable=C0411

from jaxppo.ppo.ppo import PPO
from jaxppo.ppo.train_ppo import make_train
from jaxppo.wandb_logging import LoggingConfig

NUM_ENVS = 2
NUM_STEPS = 4


def test_trained_ppo_pre_defined_no_LSTM():
    """Test that ppo init and train work on pre-defined gymnax env"""
    num_envs = NUM_ENVS
    num_steps = NUM_STEPS
    total_timesteps = NUM_ENVS * NUM_STEPS * 2

    learning_rate = 2.5e-4
    base_env, env_params = gymnax.make("CartPole-v1")
    env_id = base_env
    PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        num_minibatches=num_envs,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "32", "tanh"],
        critic_architecture=["64", "tanh", "32", "tanh"],
        env_params=env_params,
    )
    make_train(
        total_timesteps,
        num_steps,
        num_envs,
        env_id,
        learning_rate,
        env_params=env_params,
    )


def test_trained_ppo_pre_defined_env():
    """Test that ppo init and train work on pre-defined gymnax env"""
    num_envs = NUM_ENVS
    num_steps = NUM_STEPS
    total_timesteps = NUM_ENVS * NUM_STEPS * 2

    learning_rate = 2.5e-4
    base_env, env_params = gymnax.make("CartPole-v1")
    env_id = base_env
    PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "32", "tanh"],
        critic_architecture=["64", "tanh", "32", "tanh"],
        env_params=env_params,
    )
    make_train(
        total_timesteps,
        num_steps,
        num_envs,
        env_id,
        learning_rate,
        env_params=env_params,
    )


def test_trained_ppo_pre_defined_wrapped_env():
    """Test that ppo init and train work on pre-defined gymnax wrapped-env"""
    num_envs = NUM_ENVS
    num_steps = NUM_STEPS
    total_timesteps = NUM_ENVS * NUM_STEPS * 2

    learning_rate = 2.5e-4
    base_env, env_params = gymnax.make("CartPole-v1")
    wrapped_env = FlattenObservationWrapper(base_env)
    env_id = wrapped_env
    PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "32", "tanh"],
        critic_architecture=["64", "tanh", "32", "tanh"],
        env_params=env_params,
    )
    make_train(
        total_timesteps,
        num_steps,
        num_envs,
        env_id,
        learning_rate,
        env_params=env_params,
    )


# def test_ppo_test_fails_without_agent_state():
#     """Test that the ppo train function fails"""
#     num_envs = 4
#     total_timesteps = int(1e2)
#     num_steps = 8
#     learning_rate = 2.5e-4
#     env_id = "CartPole-v1"
#     agent = PPO(
#         total_timesteps=total_timesteps,
#         num_steps=num_steps,
#         num_envs=num_envs,
#         env_id=env_id,
#         learning_rate=learning_rate,
#         actor_architecture=["64", "tanh", "64", "tanh"],
#         critic_architecture=["64", "tanh", "64", "tanh"],
#     )
#     with pytest.raises(ValueError):
#         agent.test(seed=42, n_episodes=10)


def test_ppo_train():
    """Test that the ppo train function doesn't fail"""
    num_envs = NUM_ENVS
    num_steps = NUM_STEPS
    total_timesteps = NUM_ENVS * NUM_STEPS * 2

    learning_rate = 2.5e-4
    env_id = "CartPole-v1"
    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        num_minibatches=num_envs,
        learning_rate=learning_rate,
        actor_architecture=["4", "tanh"],
        critic_architecture=["4", "tanh"],
        lstm_hidden_size=2,
    )
    agent.train(seed=42, test=False)


@pytest.mark.slow
def test_ppo_train_and_log():
    """Test that the ppo train function doesn't fail"""
    num_envs = NUM_ENVS
    num_steps = NUM_STEPS
    total_timesteps = NUM_ENVS * NUM_STEPS * 2

    learning_rate = 2.5e-4
    env_id = "CartPole-v1"
    fake_logging_config = LoggingConfig("Test multithreading", "test", config={})
    folder = "wandb_fake"
    os.makedirs(folder, exist_ok=True)
    wandb.init(mode="disabled", dir=folder)
    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        num_minibatches=num_envs,
        learning_rate=learning_rate,
        actor_architecture=["4", "tanh"],
        critic_architecture=["4", "tanh"],
        lstm_hidden_size=2,
        logging_config=fake_logging_config,
    )
    agent.train(seed=42, test=False)
    wandb.finish()


# def test_ppo_train_test():
#     """Test that the ppo train function doesn't fail"""
#     num_envs = 4
#     num_steps = 4
#     total_timesteps = int(num_envs * num_steps * 2)
#     learning_rate = 2.5e-4
#     env_id = "CartPole-v1"
#     agent = PPO(
#         total_timesteps=total_timesteps,
#         num_steps=num_steps,
#         num_envs=num_envs,
#         env_id=env_id,
#         learning_rate=learning_rate,
#         actor_architecture=["4", "tanh"],
#         critic_architecture=["4", "tanh"],
#         lstm_hidden_size=2,
#     )
#     agent.train(seed=42, test=True)


def test_ppo_fails_init_with_incorrect_env():
    """Check that giving an environmen whichi is not a wrapped or unwrapped gymnax \
        env fails"""
    with pytest.raises(ValueError):
        num_envs = NUM_ENVS
        num_steps = NUM_STEPS
        total_timesteps = NUM_ENVS * NUM_STEPS * 2

        learning_rate = 2.5e-4
        env_id = gym.make("CartPole-v1")
        PPO(
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            num_envs=num_envs,
            env_id=env_id,
            learning_rate=learning_rate,
            actor_architecture=["64", "tanh"],
            critic_architecture=["64", "tanh"],
        )


def test_ppo_fails_init_with_no_env_params_in_pre_defined_env():
    """Check that using a predefined env and not providing env params fails"""
    with pytest.raises(ValueError):
        num_envs = 4
        total_timesteps = int(1e2)
        num_steps = 8
        learning_rate = 2.5e-4
        env_id, _ = gymnax.make("CartPole-v1")
        PPO(
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            num_envs=num_envs,
            env_id=env_id,
            learning_rate=learning_rate,
            actor_architecture=["64", "tanh"],
            critic_architecture=["64", "tanh"],
        )


def test_ppo_fails_init_with_wrong_env_params():
    """Check that giving bad env parameters returns an error"""
    with pytest.raises(ValueError):
        num_envs = NUM_ENVS
        num_steps = NUM_STEPS
        total_timesteps = NUM_ENVS * NUM_STEPS * 2

        learning_rate = 2.5e-4
        env_id, _ = gymnax.make("CartPole-v1")
        PPO(
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            num_envs=num_envs,
            env_id=env_id,
            learning_rate=learning_rate,
            actor_architecture=["64", "tanh"],
            critic_architecture=["64", "tanh"],
            env_params=False,
        )
