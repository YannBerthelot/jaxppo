from typing import Optional

import jax
import jax.numpy as jnp
from gymnax.environments.environment import EnvParams

from jaxppo.environment.environment_interaction import reset_env, step_env
from jaxppo.networks.networks import get_pi
from jaxppo.networks.networks_RNN import init_hidden_state
from jaxppo.types_rnn import HiddenState
from jaxppo.utils import check_env_is_gymnax


def evaluate(
    env,
    actor_state,
    num_episodes: int,
    rng: jax.Array,
    env_params: Optional[EnvParams],
    recurrent: bool = False,  # TODO : find how to infer this
    lstm_hidden_size: Optional[int] = None,  # TODO : find how to infer this
) -> jax.Array:
    """Evaluate the agent over n_episodes (using seed for rng) and log the episodic\
              returns"""
    key = rng
    mode = "gymnax" if check_env_is_gymnax(env) else "brax"
    key, init_hidden_key, reset_key = jax.random.split(key, 3)
    init_hidden_keys = jax.random.split(init_hidden_key, num_episodes)
    reset_keys = (
        jax.random.split(reset_key, num_episodes) if mode == "gymnax" else reset_key
    )

    def get_action_and_entropy(
        obs: jax.Array,
        key: jax.Array,
        hidden: Optional[HiddenState] = None,
        done: Optional[bool] = None,
    ) -> tuple[jax.Array, Optional[HiddenState], jax.Array]:
        """Returns a numpy action compliant with gym using the current \
                state of the agent"""
        if actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )
        pi, new_hidden = get_pi(
            actor_state.state,
            actor_state.state.params,
            obs[jnp.newaxis, :] if recurrent else obs,
            hidden if recurrent else None,  # shape : (1,num_envs,obs_size)
            (  # shape : (num_envs, 2)
                done.reshape(1, -1) if recurrent else None
            ),  # shape : (1,num_envs)
            recurrent,
        )
        action = pi.sample(seed=key)
        return action, new_hidden, pi.entropy()

    obs, state = reset_env(reset_keys, env, mode, env_params)
    done = jnp.zeros(num_episodes, dtype=jnp.int8)
    rewards = jnp.zeros(num_episodes)
    entropy_collected = jnp.zeros(1)
    hidden = (
        init_hidden_state(
            lstm_hidden_size,
            num_envs=num_episodes,
            rng=init_hidden_keys,
        )
        if recurrent
        else None
    )
    carry = (rewards, key, obs, done, hidden, state, entropy_collected)

    def sample_action_and_step_env(carry):
        (rewards, rng, obs, done, hidden, state, entropy_collected) = carry
        rng, action_key, step_key = jax.random.split(rng, num=3)
        step_keys = (
            jax.random.split(step_key, num_episodes) if mode == "gymnax" else step_key
        )
        actions, hidden, entropy = get_action_and_entropy(
            obs,
            action_key,
            hidden if recurrent else None,
            done if recurrent else None,
        )
        obs, state, reward, new_done, _ = step_env(
            step_keys,
            state,
            actions.squeeze(0) if recurrent else actions,
            env,
            mode,
            env_params,
        )
        entropy_collected += (
            entropy.mean() * (1 - done)
        ).mean()  # only add entropies for unfinished envs
        done = done | jnp.int8(new_done)
        rewards = rewards + (
            reward * (1 - done)
        )  # only add rewards for unfinished envs
        # int("{x} {y} {z}", x=rewards, y=reward, z=done)
        return (rewards, rng, obs, done, hidden, state, entropy_collected)

    def env_not_done(carry):
        done = carry[3]
        return jnp.logical_not(done.all())

    episodic_reward_sum, _, _, _, _, _, entropy_collected = jax.lax.while_loop(
        env_not_done, sample_action_and_step_env, carry
    )

    return episodic_reward_sum, entropy_collected
