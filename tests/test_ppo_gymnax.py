# pylint: disable = missing-function-docstring, missing-module-docstring
from math import log

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxppo.buffer import Buffer, init_buffer
from jaxppo.networks import get_adam_tx, init_actor_and_critic_state, init_networks
from jaxppo.ppo_gymnax import (
    PPO,
    get_logprob_and_action,
    ppo_actor_loss,
    ppo_critic_loss,
    predict_action_and_value_then_update_buffer,
)
from jaxppo.utils import (
    get_env_action_shape,
    get_env_observation_shape,
    make_gymnax_env,
)


@pytest.fixture
def setup_buffer():
    n_steps = 4
    n_envs = 4
    train_env, train_env_params, _ = make_gymnax_env("CartPole-v1", seed=42)
    action_space_shape = get_env_action_shape(train_env)
    observation_space_shape = get_env_observation_shape(train_env, train_env_params)
    return (
        init_buffer(
            num_steps=n_steps,
            num_envs=n_envs,
            action_space_shape=action_space_shape,
            observation_space_shape=observation_space_shape,
        ),
        train_env,
        train_env_params,
        n_steps,
        n_envs,
    )


@pytest.fixture
def setup_agent_state():
    env, env_params, _ = make_gymnax_env("CartPole-v1", seed=42)
    actor_architecture = ["32", "tanh", "32", "tanh"]
    critic_architecture = ["32", "relu", "32", "relu"]
    actor, critic = init_networks(env, actor_architecture, critic_architecture)

    key = random.PRNGKey(42)
    key, actor_key, critic_key = random.split(key, num=3)
    tx = get_adam_tx()
    actor_state, critic_state = init_actor_and_critic_state(
        actor_network=actor,
        critic_network=critic,
        actor_key=actor_key,
        critic_key=critic_key,
        env=env,
        tx=tx,
        env_params=env_params,
    )
    return actor_state, critic_state, key


def test_get_logprob_and_action():
    logits = jnp.array([-jnp.inf, -jnp.inf, 1.0, -jnp.inf])
    key = random.PRNGKey(42)
    log_prob, action, key = get_logprob_and_action(key, logits)
    assert action == 2
    assert log_prob == log(1)


def test_predict_action_and_value_then_update_buffer(setup_agent_state, setup_buffer):
    actor_state, critic_state, key = setup_agent_state
    buffer, envs, envs_params, _, num_envs = setup_buffer
    obs, _ = envs.reset(key, envs_params)
    step = 0
    buffer, action, key = predict_action_and_value_then_update_buffer(
        actor_state=actor_state,
        critic_state=critic_state,
        buffer=buffer,
        obs=jnp.array([obs for _ in range(num_envs)]),
        key=key,
        step=step,
    )
    assert isinstance(buffer, Buffer)
    assert jnp.array_equal(buffer.dones[0], jnp.zeros(num_envs))
    assert len(action) == num_envs


def test_rollout(setup_agent_state, setup_buffer):  # pylint: disable=R0914
    actor_state, critic_state, key = setup_agent_state
    buffer, envs, env_params, num_steps, num_envs = setup_buffer

    done = np.array([False for _ in range(num_envs)])

    agent = PPO(
        seed=42, num_envs=num_envs, num_steps=num_steps, env=envs, env_params=env_params
    )
    vmap_reset = jax.vmap(envs.reset, in_axes=(0, None))
    vmap_keys = jax.random.split(key, num_envs)
    obs, env_state = vmap_reset(vmap_keys, env_params)
    old_advantages = buffer.advantages
    _, _, buffer, key, envs, env_state = agent.rollout(
        actor_state=actor_state,
        critic_state=critic_state,
        env=envs,
        num_steps=num_steps,
        obs=obs,
        done=done,
        buffer=buffer,
        key=key,
        env_state=env_state,
        step_key=vmap_keys,
    )
    assert jnp.array_equal(buffer.obs[0], obs)
    assert jnp.array_equal(buffer.advantages, old_advantages)


def test_train():
    num_envs = 2
    num_steps = 2
    train_env, train_env_params, _ = make_gymnax_env("CartPole-v1", seed=42)
    agent = PPO(
        seed=42,
        num_envs=num_envs,
        num_steps=num_steps,
        env=train_env,
        env_params=train_env_params,
    )
    old_actor_params = agent.actor_state.params
    old_critic_params = agent.critic_state.params
    agent.train(env=train_env, total_timesteps=int(16))
    assert not jnp.array_equal(agent.actor_state.params, old_actor_params)
    assert not jnp.array_equal(agent.critic_state.params, old_critic_params)


def test_ppo_actor_loss_does_not_fail(setup_agent_state, setup_buffer):
    actor_state, _, key = setup_agent_state
    _, envs, env_params, _, num_envs = setup_buffer
    obs, _ = envs.reset(key, env_params)
    action = jnp.array([envs.action_space().sample(key) for _ in range(num_envs)])
    logprob = jnp.array([0.5 for _ in range(num_envs)])
    step = (obs, action, logprob, logprob, logprob, logprob)
    ppo_actor_loss(
        actor_state=actor_state,
        actor_params=actor_state.params,
        trajectory_and_variables=step,
        clip_coef=0.1,
        ent_coef=0.01,
    )


def test_ppo_critic_loss_does_not_fail(setup_agent_state, setup_buffer):
    _, critic_state, key = setup_agent_state
    _, envs, env_params, _, num_envs = setup_buffer
    obs, _ = envs.reset(key, env_params)
    action = envs.action_space().sample(key)
    logprob = jnp.array([0.5 for _ in range(num_envs)])
    step = (obs, action, logprob, logprob, logprob, logprob)
    ppo_critic_loss(
        critic_state=critic_state,
        critic_params=critic_state.params,
        trajectory_and_variables=step,
    )


# def test_update_ppo_does_not_fail(setup_agent_state, setup_buffer):
#     agent_state, key = setup_agent_state
#     buffer, envs, num_steps, num_envs = setup_buffer
#     batch_size = num_envs * num_steps
#     total_timesteps = 100
#     num_updates = total_timesteps // batch_size
#     obs, _ = envs.reset()
#     done = np.array([False for _ in range(num_envs)])
#     gamma = 0.9
#     gae_lambda = 0.5

#     for update in range(1, num_updates + 1):
#         new_obs, new_done, buffer, key, env = rollout(
#             agent_state=agent_state,
#             env=envs,
#             num_steps=num_steps,
#             obs=obs,
#             done=done,
#             buffer=buffer,
#             key=key,
#         )
#         next_value = predict_value(agent_state, agent_state.params, new_obs).squeeze()
#         buffer = update_gae_advantages(buffer, new_done, next_value, gamma, gae_lambda)
#         buffer = update_returns(buffer)

#     update_ppo(agent_state=agent_state, buffer=buffer, key=key, batch_size=batch_size)
