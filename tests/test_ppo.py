"""Test ppo"""
import pytest
import wandb

from jaxppo.ppo_pure_rl import PPO

wandb.init(mode="disabled")


def test_ppo_train():
    """Test that the ppo train function doesn't fail"""
    num_envs = 8
    total_timesteps = int(1e4)
    num_steps = 128
    learning_rate = 2.5e-4
    env_id = "CartPole-v1"
    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        log=False,
    )
    agent.train(seed=42, test=False)


def test_ppo_test_fails_without_agent_state():
    """Test that the ppo train function doesn't fail"""
    num_envs = 8
    total_timesteps = int(1e4)
    num_steps = 128
    learning_rate = 2.5e-4
    env_id = "CartPole-v1"
    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        log=False,
    )
    with pytest.raises(ValueError):
        agent.test(seed=42, n_episodes=10)


def test_ppo_train_and_test():
    """Test that the ppo train function doesn't fail"""
    num_envs = 8
    total_timesteps = int(1e4)
    num_steps = 128
    learning_rate = 2.5e-4
    env_id = "CartPole-v1"
    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        log=False,
    )
    agent.train(seed=42, test=True)


def test_ppo_train_and_log():
    """Test that the ppo train function doesn't fail"""
    num_envs = 8
    total_timesteps = int(1e4)
    num_steps = 128
    learning_rate = 2.5e-4
    env_id = "CartPole-v1"
    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        log=True,
    )
    agent.train(seed=42, test=True)
