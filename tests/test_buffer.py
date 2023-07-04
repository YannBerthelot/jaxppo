import pdb

import gymnasium as gym
import jax.numpy as jnp
import pytest

from jaxppo.buffer import init_buffer, insert_buffer, update_gae_advantages


def get_env_action_shape(env: gym.Env):
    return (
        (env.action_space.n,)
        if isinstance(env.action_space, gym.spaces.Discrete)
        else env.action_space.shape
    )


@pytest.fixture
def setup_buffer():
    n_steps = 4
    n_envs = 2
    env = gym.make("CartPole-v1")
    action_space_shape = get_env_action_shape(env)
    observation_space_shape = env.observation_space.shape
    return init_buffer(
        num_steps=n_steps,
        num_envs=n_envs,
        action_space_shape=action_space_shape,
        observation_space_shape=observation_space_shape,
    )


def test_init_buffer(setup_buffer):
    buffer = setup_buffer
    n_steps = 4
    n_envs = 2
    env = gym.make("CartPole-v1")
    action_space_shape = get_env_action_shape(env)
    observation_space_shape = env.observation_space.shape

    assert buffer.dones.shape == (n_steps, n_envs)
    assert buffer.actions.shape == (n_steps, n_envs, *action_space_shape)
    assert buffer.obs.shape == (n_steps, n_envs, *observation_space_shape)
    assert jnp.array_equal(buffer.dones, jnp.zeros(buffer.dones.shape))


def test_insert_buffer(setup_buffer):
    buffer = setup_buffer
    step = 1
    feature_dict = {"rewards": jnp.ones(2), "advantages": jnp.ones(2) * 2}
    buffer = insert_buffer(buffer, step, feature_dict)
    assert jnp.allclose(buffer.rewards[1], jnp.ones(2))
    assert jnp.allclose(buffer.advantages[1], jnp.ones(2) * 2)


def test_compute_gae():
    dones = [0, 0, 0, 0]
    values = [0.1, 0.1, 0.1, 0.1]
    rewards = [1.0, 1.0, 1.0, 1.0]
    next_value = 0.5
    next_done = 1
    gamma = 0.9
    gae_lambda = 0.9

    # Compute "flat" advantages
    advantage_4 = rewards[-1] + gamma * next_value * (1 - next_done) - values[-1]
    advantage_3 = rewards[-2] + gamma * values[-1] * (1 - dones[-1]) - values[-2]
    advantage_2 = rewards[-3] + gamma * values[-2] * (1 - dones[-2]) - values[-3]
    advantage_1 = rewards[-4] + gamma * values[-3] * (1 - dones[-3]) - values[-4]

    # Compute GAE/exponential average
    gae_4 = advantage_4
    gae_3 = advantage_3 + gae_lambda * gamma * gae_4
    gae_2 = advantage_2 + gae_lambda * gamma * gae_3
    gae_1 = advantage_1 + gae_lambda * gamma * gae_2

    n_steps = 4
    n_envs = 1
    env = gym.make("CartPole-v1")
    action_space_shape = get_env_action_shape(env)
    observation_space_shape = env.observation_space.shape

    buffer = init_buffer(
        num_steps=n_steps,
        num_envs=n_envs,
        action_space_shape=action_space_shape,
        observation_space_shape=observation_space_shape,
    )
    for step in range(4):
        buffer = insert_buffer(
            buffer,
            step,
            {"dones": dones[step], "rewards": rewards[step], "values": values[step]},
        )
    buffer = update_gae_advantages(buffer, next_done, next_value, gamma, gae_lambda)
    assert jnp.allclose(
        buffer.advantages, jnp.array([[gae_1], [gae_2], [gae_3], [gae_4]])
    )
