"""Test ppo"""

import gymnasium as gym
import gymnax
import jax.numpy as jnp
import numpy as np
import pytest
import wandb
from gymnax.wrappers.purerl import FlattenObservationWrapper  # pylint: disable=C0411

from jaxppo.ppo import PPO
from jaxppo.train import Transition, _calculate_gae, make_train
from jaxppo.wandb_logging import LoggingConfig


def test_compute_gae():
    """Check that the computation of gae matches with sb3"""
    dones = [False, False, True]
    values = [1.0, 1.0, 1.0]
    rewards = [1.0, 1.0, 1.0]
    traj_batch = Transition(
        done=jnp.array(dones),
        action=jnp.array(dones),
        value=jnp.array(values),
        reward=jnp.array(rewards),
        log_prob=jnp.array(dones),
        obs=jnp.array(dones),
        info=jnp.array(dones),
    )
    dones = [False, False, False]
    last_val = jnp.array(0.5)
    last_done = True
    gamma = 0.5
    gae_lambda = 0.95
    advantages, targets = _calculate_gae(
        traj_batch, last_val, gamma=gamma, gae_lambda=gae_lambda
    )
    ##### from sb3 #####
    last_gae_lam = 0
    buffer_size = 3
    exepected_advantages = np.zeros(buffer_size)
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - last_done
            next_values = last_val
        else:
            next_non_terminal = 1.0 - dones[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        exepected_advantages[step] = last_gae_lam
    expected_targets = exepected_advantages + values
    ##### from sb3 #####
    assert jnp.allclose(advantages, exepected_advantages)
    assert jnp.allclose(targets, expected_targets)


def test_trained_ppo_pre_defined_env():
    """Test that ppo init and train work on pre-defined gymnax env"""
    num_envs = 4
    total_timesteps = int(1e2)
    num_steps = 8
    learning_rate = 2.5e-4
    base_env, env_params = gymnax.make("CartPole-v1")
    env_id = base_env
    PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
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
    num_envs = 4
    total_timesteps = int(1e2)
    num_steps = 8
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
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
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


def test_ppo_train():
    """Test that the ppo train function doesn't fail"""
    num_envs = 4
    total_timesteps = int(1e2)
    num_steps = 8
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
    )
    agent.train(seed=42, test=False)


def test_ppo_test_fails_without_agent_state():
    """Test that the ppo train function doesn't fail"""
    num_envs = 4
    total_timesteps = int(1e2)
    num_steps = 8
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
    )
    with pytest.raises(ValueError):
        agent.test(seed=42, n_episodes=10)


def test_ppo_train_and_test_shared_network():
    """Test that the ppo train function doesn't fail"""
    num_envs = 2
    num_steps = 4
    total_timesteps = int(num_envs * num_steps * 2)
    learning_rate = 2.5e-4
    env_id = "CartPole-v1"
    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "64", "tanh"],
        shared_network=True,
        vf_coef=0.5,
    )
    agent.train(seed=42, test=True)


def test_ppo_train_and_test():
    """Test that the ppo train function doesn't fail"""
    num_envs = 2
    num_steps = 4
    total_timesteps = int(num_envs * num_steps * 2)
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
    )
    agent.train(seed=42, test=True)


def test_ppo_train_and_log():
    """Test that the ppo train function doesn't fail"""
    num_envs = 4
    total_timesteps = int(1e2)
    num_steps = 8
    learning_rate = 2.5e-4
    env_id = "CartPole-v1"

    try:
        logging_config = LoggingConfig("Test", "test", {"test": 1}, mode="disabled")
        agent = PPO(
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            num_envs=num_envs,
            env_id=env_id,
            learning_rate=learning_rate,
            actor_architecture=["64", "tanh", "64", "tanh"],
            critic_architecture=["64", "tanh", "64", "tanh"],
            logging_config=logging_config,
        )
        agent.train(seed=42, test=True)
    finally:
        wandb.finish()


def test_ppo_fails_init_with_incorrect_env():
    """Check that giving an environmen whichi is not a wrapped or unwrapped gymnax \
        env fails"""
    with pytest.raises(ValueError):
        num_envs = 4
        total_timesteps = int(1e2)
        num_steps = 8
        learning_rate = 2.5e-4
        env_id = gym.make("CartPole-v1")
        PPO(
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            num_envs=num_envs,
            env_id=env_id,
            learning_rate=learning_rate,
            actor_architecture=["64", "tanh", "64", "tanh"],
            critic_architecture=["64", "tanh", "64", "tanh"],
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
            actor_architecture=["64", "tanh", "64", "tanh"],
            critic_architecture=["64", "tanh", "64", "tanh"],
        )


def test_ppo_fails_init_with_wrong_env_params():
    """Check that giving bad env parameters returns an error"""
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
            actor_architecture=["64", "tanh", "64", "tanh"],
            critic_architecture=["64", "tanh", "64", "tanh"],
            env_params=False,
        )
