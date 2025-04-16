"""PPO class to easily handle the agent"""

from typing import Optional

import jax
import jax.numpy as jnp
from gymnax.environments.environment import (  # TODO : mix brax here
    Environment,
    EnvParams,
)
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import random

from jaxppo.evaluate import evaluate
from jaxppo.networks.networks import get_pi
from jaxppo.networks.networks import predict_probs as network_predict_probs
from jaxppo.networks.networks import predict_value as network_predict_value
from jaxppo.ppo.ppo_config import PPOConfig
from jaxppo.ppo.train_ppo import init_agent, make_train
from jaxppo.types_rnn import HiddenState
from jaxppo.utils import build_env_from_id, load_model, prepare_env
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
        save: bool = False,
        save_folder: str = "./models",
        log_video: bool = False,
        video_log_frequency: Optional[int] = None,
        save_frequency: Optional[int] = None,
        lstm_hidden_size: Optional[int] = None,
        seed: int = 42,
        average_reward: bool = False,
        window_size: int = 32,
        episode_length: Optional[int] = None,
        render_env_id: Optional[str] = None,
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

        key = random.PRNGKey(seed)
        num_updates = total_timesteps // num_steps // num_envs

        env, env_params, env_id, continuous = prepare_env(
            env_id, env_params=env_params, num_envs=num_envs
        )  # TODO : make it simpler as it is only used for initializing the networks

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
            save=save,
            save_folder=save_folder,
            log_video=log_video,
            log_video_frequency=video_log_frequency,
            save_frequency=save_frequency,
            lstm_hidden_size=lstm_hidden_size,
            continuous=continuous,
            average_reward=average_reward,
            window_size=window_size,
            episode_length=episode_length,
            render_env_id=render_env_id,
        )

        (
            self._actor_state,
            self._critic_state,
            self.rng,
        ) = init_agent(
            key,
            env,
            actor_architecture,
            critic_architecture,
            num_envs,
            anneal_lr,
            learning_rate,
            num_minibatches,
            update_epochs,
            num_updates,
            max_grad_norm,
            env_params,
            continuous=continuous,
        )
        self.recurrent = lstm_hidden_size is not None

    def load(self, path: str):
        """Load the actor and critic state from the save file in path"""
        self._actor_state, self._critic_state = load_model(
            path, self._actor_state, self._critic_state
        )

    def train(self, seed: int, test: bool = False, wandb_folder: Optional[str] = None):
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
            init_logging(self.config.logging_config, folder=wandb_folder)

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
            save=self.config.save,
            save_folder=self.config.save_folder,
            save_frequency=self.config.save_frequency,
            video_log_frequency=self.config.log_video_frequency,
            log_video=self.config.log_video,
            lstm_hidden_size=self.config.lstm_hidden_size,
            continuous=self.config.continuous,
            average_reward_mode=self.config.average_reward,
            window_size=self.config.window_size,
            episode_length=self.config.episode_length,
            render_env_id=self.config.render_env_id,
        )

        runner_state = train_jit(key)
        self._actor_state, self._critic_state = (
            runner_state.actor_state,
            runner_state.critic_state,
        )
        self._actor_hidden_state, self._critic_hidden_state = (
            runner_state.actor_state.hidden_state,
            runner_state.critic_state.hidden_state,
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
        if self._actor_hidden_state is None and self.recurrent:
            raise ValueError("Actor hidden state is not defined in reccurent mode")

        probs = network_predict_probs(
            self._actor_state.state,
            self._actor_state.state.params,
            obs[jnp.newaxis, :] if self.recurrent else obs,
            self._actor_hidden_state[0][jnp.newaxis, :] if self.recurrent else None,  # type: ignore[index]
            (jnp.array([[True]]) if self.recurrent else None),  # type: ignore[index]
            recurrent=self.recurrent,
        )
        return probs[0][0] if self.recurrent else probs

    def predict_value(
        self,
        obs: jax.Array,
        hidden: Optional[HiddenState] = None,
        done: Optional[bool] = None,
    ) -> tuple[jax.Array, Optional[HiddenState]]:
        """Predict the value of obs according to critic"""
        if self._critic_state is None:
            raise ValueError(
                "Attempted to predict value without a critic state/training the agent"
                " first"
            )
        if hidden is None and self.recurrent:
            if self._critic_hidden_state is None:
                raise ValueError("Critic hidden state is not defined in reccurent mode")
            hidden = self._critic_hidden_state[0][jnp.newaxis, :]

        return network_predict_value(
            self._critic_state.state,
            self._critic_state.state.params,
            obs[jnp.newaxis, :] if self.recurrent else obs,
            hidden if self.recurrent else None,
            jnp.array([[done]]) if self.recurrent else None,
            recurrent=self.recurrent,
        )

    def get_action(
        self,
        obs: jax.Array,
        key: jax.Array,
        hidden: Optional[HiddenState] = None,
        done: Optional[bool] = None,
    ) -> tuple[jax.Array, Optional[HiddenState]]:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        if self._actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )

        # pi = self._actor_state.apply_fn(self._actor_state.params, obs)
        pi, new_hidden = get_pi(
            self._actor_state.state,
            self._actor_state.state.params,
            obs[jnp.newaxis, :] if self.recurrent else obs,
            hidden if self.recurrent else None,
            done if self.recurrent else None,
            self.recurrent,
        )
        action = pi.sample(seed=key)
        return action, new_hidden

    def get_action_distribution(
        self,
        obs: jax.Array,
        hidden: Optional[HiddenState] = None,
        done: Optional[bool] = None,
    ) -> tuple[jax.Array, Optional[HiddenState]]:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        if self._actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )

        # pi = self._actor_state.apply_fn(self._actor_state.params, obs)
        pi, _ = get_pi(
            self._actor_state,
            self._actor_state.params,
            obs[jnp.newaxis, :] if self.recurrent else obs,
            hidden if self.recurrent else None,
            done if self.recurrent else None,
            self.recurrent,
        )
        return pi

    def test(self, num_episodes: int, seed: int):
        """Evaluate the agent over n_episodes (using seed for rng) and log the episodic\
              returns"""
        key = random.PRNGKey(seed)
        if isinstance(self.config.env_id, str):
            env, env_params = build_env_from_id(self.config.env_id)
        else:  # env is assumed to be provided already built
            env, env_params = self.config.env_id, self.config.env_params

        episodic_reward_sum, _ = evaluate(
            env,
            self._actor_state,
            num_episodes,
            key,
            env_params,
            recurrent=self.recurrent,
            lstm_hidden_size=self.config.lstm_hidden_size,
        )

        if self.config.logging_config is not None:
            wandb_test_log(episodic_reward_sum)


env_dict = {
    "HalfCheetah-v4": "halfcheetah",
    "Hopper-v4": "hopper",
    "Humanoid-v4": "humanoid",
    "HumanoidStandup-v4": "humanoidstandup",
    "InvertedDoublePendulum-v4": "inverted_double_pendulum",
    "InvertedPendulum-v4": "inverted_pendulum",
    "Pusher-v4": "pusher",
    "Reacher-v4": "reacher",
    "Swimmer-v4": "swimmer",
    "Walker2d-v4": "walker",
}

if __name__ == "__main__":
    import wandb

    true_env_id = "HalfCheetah-v4"
    num_envs = 4
    num_steps = 2048
    env_id = env_dict[true_env_id] if true_env_id in env_dict.keys() else true_env_id
    logging_config = LoggingConfig("Continuous PPO", "test", config={})
    init_logging(logging_config=logging_config, folder=None)
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
        max_grad_norm=0.5,
        save=True,
        log_video=True,
        video_log_frequency=None,
        average_reward=False,
        render_env_id=true_env_id,
        # window_size=2048,
        # lstm_hidden_size=16,
    )

    def training_loop(seed):
        """Train the agent in a functional fashion for jax"""
        agent.train(seed, test=False)
        wandb.finish()

    # training_loop(42)
    seeds = jnp.array(range(2))
    jax.vmap(training_loop, in_axes=(0))(seeds)
