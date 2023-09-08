"""PPO class to easily handle the agent"""
from typing import Optional

import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, EnvParams
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import random

from jaxppo.config import PPOConfig
from jaxppo.networks.networks_RNN import ScannedRNN, init_hidden_state
from jaxppo.networks.networks_RNN import predict_probs as network_predict_probs
from jaxppo.networks.networks_RNN import (
    predict_value_and_hidden as network_predict_value_and_hidden,
)
from jaxppo.train_rnn import make_train
from jaxppo.wandb_logging import LoggingConfig, init_logging, wandb_test_log


class PPO:
    """PPO Agent that allows simple training and testing"""

    def __init__(  # pylint: disable=W0102, R0913
        self,
        total_timesteps: int,
        num_steps: int,
        num_envs: int,
        env_id: str | Environment | GymnaxWrapper,
        learning_rate: float,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        clip_coef_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        logging_config: Optional[LoggingConfig] = None,
        env_params: Optional[EnvParams] = None,
        anneal_lr: bool = True,
        max_grad_norm: float = 0.5,
        advantage_normalization: bool = True,
        lstm_hidden_size: int = 8,
    ) -> None:
        """
        PPO Agent that allows simple training and testing

        Args:
        total_timesteps (int): Total number of timesteps (distributed across all envs)\
              to train for.
        num_steps (int): Number of steps to run per environment before updating.
        num_envs (int): Number of environments to run in parrallel.
        env_id (str | Environment | GymnaxWrapper): The gym-id of the environment to\
              use or a pre-defined wrapped or unwrapped gymnax env .
        learning_rate (float): The learning rate for the optimizer of networks.
        num_minibatches (int, optional): The number of minibatches (number of shuffled\
              minibatch in an epoch). Defaults to 4.
        update_epochs (int, optional): The number of epochs to run on the same\
              collected data. Defaults to 4.
        actor_architecture (list, optional): The description of the architecture of the\
              actor network. Defaults to ["64", "tanh", "64", "tanh"].
        critic_architecture (list, optional): The description of the architecture of\
              the critic network. Defaults to ["64", "relu", "relu", "tanh"].
        gamma (float, optional): The discount factor. Defaults to 0.99.
        gae_lambda (float, optional): The gae advantage computation lambda parameter.\
             Defaults to 0.95.
        clip_coef (float, optional): PPO's clipping coefficient. Defaults to 0.2.
        ent_coef (float, optional): The entropy coefficient in the actor loss.\
              Defaults to 0.01.
        log (bool, optional): Wether or not to log training using wandb.
        """

        self.config = PPOConfig(
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            num_envs=num_envs,
            env_id=env_id,
            learning_rate=learning_rate,
            num_minibatches=num_minibatches,
            update_epochs=update_epochs,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_coef=clip_coef,
            clip_coef_vf=clip_coef_vf,
            ent_coef=ent_coef,
            logging_config=logging_config,
            env_params=env_params,
            anneal_lr=anneal_lr,
            max_grad_norm=max_grad_norm,
            advantage_normalization=advantage_normalization,
            lstm_hidden_size=lstm_hidden_size,
        )

        self._actor_state = None
        self._critic_state = None
        self._actor_hidden_state = None
        self._critic_hidden_state = None
        self.actor_network = None
        self.critic_network = None

    def train(self, seed: int, test: bool = False):
        """
        Trains the agent using the agent's config, will also evaluate at the end if test is set to true

        Args:
            seed (int): The seed for the agent's rng
            test (bool, optional): Wether or not to evaluate the agent over test\
                  episodes at the end. Defaults to False.
        """
        key = random.PRNGKey(seed)

        if self.config.logging_config is not None:
            config = self.config.to_dict()
            config = {  # Remove logging config from config
                k: config[k] for k in set(list(config.keys())) - set(["logging_config"])
            }
            self.config.logging_config.config = dict(
                config, **self.config.logging_config.config
            )
            init_logging(self.config.logging_config)

        train_jit = make_train(
            total_timesteps=self.config.total_timesteps,
            num_steps=self.config.num_steps,
            num_envs=self.config.num_envs,
            env_id=self.config.env_id,
            learning_rate=self.config.learning_rate,
            num_minibatches=self.config.num_minibatches,
            update_epochs=self.config.update_epochs,
            actor_architecture=self.config.actor_architecture,
            critic_architecture=self.config.critic_architecture,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_coef=self.config.clip_coef,
            clip_coef_vf=self.config.clip_coef_vf,
            ent_coef=self.config.ent_coef,
            log=self.config.logging_config is not None,
            env_params=self.config.env_params,
            anneal_lr=self.config.anneal_lr,
            max_grad_norm=self.config.max_grad_norm,
            advantage_normalization=self.config.advantage_normalization,
            lstm_hidden_size=self.config.lstm_hidden_size,
        )

        runner_state = train_jit(key)
        self._actor_state, self._critic_state = (
            runner_state[0],
            runner_state[1],
        )
        self._actor_hidden_state, self._critic_hidden_state = (
            runner_state[-3],
            runner_state[-2],
        )
        if test:
            self.test(self.config.num_episode_test, seed=seed)

    def predict_probs(self, obs: jax.Array) -> jax.Array:
        """Predict the policy's probabilities of taking actions in obs"""
        if self._actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )
        return network_predict_probs(
            self._actor_state,
            self._actor_state.params,
            self._actor_hidden_state[0][jnp.newaxis, :],
            obs[jnp.newaxis, :],
            done=jnp.array([[True]]),
        )[0][0]

    def predict_value_and_hidden(
        self, obs: jax.Array, done: bool = True, hidden: Optional[jax.Array] = None
    ) -> jax.Array:
        """Predict the value of obs according to critic"""
        if self._critic_state is None:
            raise ValueError(
                "Attempted to predict value without a critic state/training the agent"
                " first"
            )
        if hidden is None:
            hidden = self._critic_hidden_state[0][jnp.newaxis, :]
        return network_predict_value_and_hidden(
            self._critic_state,
            self._critic_state.params,
            hidden,
            obs[jnp.newaxis, :],
            done=jnp.array([[done]]),
        )

    def get_action(
        self,
        obs: jax.Array,
        key: random.PRNGKeyArray,
        done: bool = True,
        hidden: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        if self._actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )
        new_hidden, pi = self._actor_state.apply_fn(
            self._actor_state.params,
            hidden,
            (obs, done),
        )
        return new_hidden, pi.sample(seed=key)

    def test(self, n_episodes: int, seed: int):
        """Evaluate the agent over n_episodes (using seed for rng) and log the episodic\
              returns"""
        key = random.PRNGKey(seed)
        (key, action_key, hidden_init_key) = random.split(key, num=3)
        if isinstance(self.config.env_id, str):
            env, env_params = gymnax.make(self.config.env_id)
        else:
            env, env_params = self.config.env_id, self.config.env_params

        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_keys = jax.random.split(key, n_episodes)

        obs, state = vmap_reset(vmap_keys, env_params)
        done = jnp.zeros(n_episodes, dtype=jnp.int8)
        rewards = jnp.zeros(n_episodes)
        hidden = init_hidden_state(
            ScannedRNN(self.config.lstm_hidden_size),
            num_envs=n_episodes,
            rng=hidden_init_key,
        )
        carry = (rewards, action_key, obs, done, hidden, state)

        def step_env(carry):
            (rewards, action_key, obs, done, hidden, state) = carry
            action_key, rng = jax.random.split(action_key)
            hidden, actions = self.get_action(
                obs[jnp.newaxis, :], rng, done=done[jnp.newaxis, :], hidden=hidden
            )
            obs, state, reward, new_done, _ = vmap_step(
                vmap_keys, state, actions.squeeze(0), env_params
            )
            done = done | jnp.int8(new_done)
            rewards = rewards + (reward * (1 - done))
            return (rewards, action_key, obs, done, hidden, state)

        def not_all_env_done(carry):
            done = carry[3]
            return jnp.logical_not(done.all())

        rewards = jax.lax.while_loop(not_all_env_done, step_env, carry)[0]

        if self.config.logging_config is not None:
            wandb_test_log(rewards)


if __name__ == "__main__":
    import wandb

    num_envs = 4
    num_steps = 8
    env_id = "CartPole-v1"
    logging_config = LoggingConfig("Test multithreading", "test", config={})

    sb3_batch_size = 128
    agent = PPO(
        env_id=env_id,
        learning_rate=2.5e-4,
        num_steps=num_steps,
        num_minibatches=num_envs,  # must be equal to num_envs atm
        update_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        logging_config=logging_config,
        total_timesteps=int(1e6),
        num_envs=num_envs,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        clip_coef_vf=None,
        anneal_lr=False,
        max_grad_norm=0.5,
        lstm_hidden_size=4,
    )

    def training_loop(seed):
        """Train the agent in a functional fashion for jax"""
        agent.train(seed, test=False)
        wandb.finish()

    seeds = jnp.array(range(5))
    jax.vmap(training_loop, in_axes=(None, 0))(seeds)