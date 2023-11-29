"""PPO class to easily handle the agent"""
from typing import Optional

import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, EnvParams
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import random

from jaxppo.config import PPOConfig
from jaxppo.networks.networks import predict_probs as network_predict_probs
from jaxppo.networks.networks import predict_probs_and_value
from jaxppo.networks.networks import predict_value as network_predict_value
from jaxppo.train import init_agent, load_model, make_train
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
        shared_network: bool = False,
        vf_coef: Optional[float] = None,
        max_grad_norm: float = 0.5,
        advantage_normalization: bool = True,
        save: bool = False,
        save_folder: str = "./models",
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
            shared_network=shared_network,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            advantage_normalization=advantage_normalization,
            save=save,
            save_folder=save_folder,
        )
        key = random.PRNGKey(0)
        num_updates = total_timesteps // num_steps // num_envs
        if isinstance(env_id, str):
            env_id, env_params = gymnax.make(env_id)
        env = env_id
        (
            self.actor_network,
            self.critic_network,
            self._actor_state,
            self._critic_state,
            self.rng,
        ) = init_agent(
            key,
            env,
            actor_architecture,
            critic_architecture,
            shared_network,
            num_envs,
            anneal_lr,
            learning_rate,
            num_minibatches,
            update_epochs,
            num_updates,
            max_grad_norm,
            env_params,
        )

    def load(self, path: str):
        """Load the actor and critic state from the save file in path"""
        self._actor_state, self._critic_state = load_model(
            path, self._actor_state, self._critic_state
        )

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
            shared_network=self.config.shared_network,
            vf_coef=self.config.vf_coef,
            advantage_normalization=self.config.advantage_normalization,
            save=self.config.save,
            save_folder=self.config.save_folder,
        )

        runner_state = train_jit(key)
        self._actor_state, self._critic_state = (
            runner_state.actor_state,
            runner_state.critic_state,
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
        if self.config.shared_network:
            return predict_probs_and_value(
                self._actor_state, self._actor_state.params, obs
            )[0]
        return network_predict_probs(self._actor_state, self._actor_state.params, obs)

    def predict_value(self, obs: jax.Array) -> jax.Array:
        """Predict the value of obs according to critic"""
        if self._critic_state is None:
            raise ValueError(
                "Attempted to predict value without a critic state/training the agent"
                " first"
            )
        if self.config.shared_network:
            return predict_probs_and_value(
                self._actor_state, self._actor_state.params, obs
            )[1]
        return network_predict_value(self._critic_state, self._critic_state.params, obs)

    def get_action(self, obs: jax.Array, key: random.PRNGKeyArray) -> jax.Array:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        if self._actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )
        if self.config.shared_network:
            pi, _ = self._actor_state.apply_fn(self._actor_state.params, obs)
        else:
            pi = self._actor_state.apply_fn(self._actor_state.params, obs)
        return pi.sample(seed=key)

    def test(self, n_episodes: int, seed: int):
        """Evaluate the agent over n_episodes (using seed for rng) and log the episodic\
              returns"""
        key = random.PRNGKey(seed)
        (
            key,
            action_key,
        ) = random.split(key, num=2)
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
        step = 0
        while not done.all():
            action_key, rng = jax.random.split(action_key)
            actions = self.get_action(obs, rng)
            obs, state, reward, new_done, _ = vmap_step(
                vmap_keys, state, actions, env_params
            )
            step += 1
            done = done | jnp.int8(new_done)
            rewards = rewards + (reward * (1 - done))
        if self.config.logging_config is not None:
            wandb_test_log(rewards)


if __name__ == "__main__":
    import wandb

    num_envs = 4
    num_steps = 2048
    env_id = "CartPole-v1"
    logging_config = LoggingConfig("Jax PPO shared", "test", config={})
    init_logging(logging_config=logging_config)
    sb3_batch_size = 64
    agent = PPO(
        env_id=env_id,
        learning_rate=3e-4,
        num_steps=num_steps,
        num_minibatches=num_steps * num_envs // sb3_batch_size,
        update_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        logging_config=logging_config,
        total_timesteps=int(1e6),
        num_envs=num_envs,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        clip_coef_vf=None,
        anneal_lr=False,
        shared_network=False,
        max_grad_norm=0.5,
        vf_coef=0.5,
        save=True,
    )

    def training_loop(seed):
        """Train the agent in a functional fashion for jax"""
        agent.train(seed, test=False)
        wandb.finish()

    training_loop(42)
    # seeds = jnp.array(range(2))
    # jax.vmap(training_loop, in_axes=(None, 0))(seeds)
