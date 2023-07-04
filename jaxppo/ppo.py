import gymnasium as gym
import jax
from jax import device_get
from numpy import ndarray
import pdb
from jaxppo.buffer import Buffer, insert_buffer
from jaxppo.networks import Network
from jax import random
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfp


def get_logprob_and_action(key: random.PRNGKeyArray, logits: jnp.array):
    key, subkey = random.split(key)
    probs = tfp.Categorical(logits)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action)
    return logprob, action


def select_action_and_update_buffer(buffer, obs, done, key, step):
    value = predict_value(obs)
    action_logits = predict_action_logits(obs)
    logprob, action = get_logprob_and_action(key, action_logits)

    buffer_step_dict = {
        "obs": obs,
        "dones": done,
        "actions": action,
        "logprobs": logprob,
        "values": value,
    }
    buffer = insert_buffer(buffer, step, buffer_step_dict)


def rollout(
    env: gym.Env,
    num_steps: int,
    obs: ndarray,
    done: bool,
    buffer: Buffer,
    key: random.PRNGKeyArray,
):
    """Rollout the policy on the environment for num_steps starting in obs/done"""
    for step in range(num_steps):
        buffer, action, key = select_action_and_update_buffer(obs, done, buffer, key)
        obs, reward, terminated, truncated, _ = env.step(device_get(action))
        done = terminated | truncated
        buffer = insert_buffer(buffer, step, {"rewards": [reward]})
    return obs, done, buffer, key
