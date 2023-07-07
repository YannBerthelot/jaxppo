"""Implementation of PPO using predefined Networks and Buffer"""
from typing import Dict, Sequence, Tuple, Union

import gymnasium as gym
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfp
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from jax import Array, device_get, random, value_and_grad
from jax.lax import stop_gradient
from numpy import ndarray
from tqdm import tqdm

from jaxppo.buffer import (
    Buffer,
    init_buffer,
    insert_buffer,
    update_gae_advantages,
    update_returns,
)
from jaxppo.networks import (
    ActivationFunction,
    AgentParams,
    AgentState,
    get_adam_tx,
    init_agent_state,
    init_networks,
    predict_action_logits,
    predict_value,
)
from jaxppo.utils import get_env_action_shape, get_env_observation_shape, make_envs
from jaxppo.wandb_logging import LoggingConfig, init_logging, log_variables


def get_logprob_and_action(
    key: random.PRNGKeyArray, logits: Array
) -> Tuple[Array, Array, random.PRNGKeyArray]:
    """Create a distribution from the logits, sample an action and compute its logprob"""
    key, subkey = random.split(key)
    probs = tfp.Categorical(logits)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action)
    return logprob, action.reshape(-1, 1) if len(action.shape) == 1 else action, key


def predict_action_and_value_then_update_buffer(
    agent_state: AgentState,
    buffer: Buffer,
    obs: ndarray,
    done: bool,
    key: random.PRNGKeyArray,
    step: int,
) -> Tuple[Buffer, Array, random.PRNGKeyArray]:
    """
    -Select an action using the current actor-network state
    -Compute the value of the current obs
    -Compute log-prob of the selected action
    -Store these values into the buffer
    """
    value = predict_value(agent_state, agent_state.params, obs)
    action_logits = predict_action_logits(agent_state, agent_state.params, obs)
    logprob, action, key = get_logprob_and_action(key, action_logits)
    buffer = insert_buffer(
        buffer,
        step,
        obs=obs,
        dones=done,
        actions=action,
        logprobs=logprob,
        values=value.squeeze(),
    )
    return buffer, action, key


def compute_logprob_entropy_value_for_update(agent_state, agent_params, obs, action):
    """Compute values needed for the PPO loss based on the agent state and \
        observed action and obs"""
    action_logits = predict_action_logits(agent_state, agent_params, obs)
    value = predict_value(agent_state, agent_params, obs)
    probs = tfp.Categorical(action_logits)
    return probs.log_prob(action), probs.entropy(), value.squeeze()


def compute_PPO_actor_loss(normalized_advantages, ratio, clip_coef):
    """Compute the actor loss according to PPO rule"""
    # Policy loss
    actor_loss_unclipped = -normalized_advantages * ratio
    actor_loss_clipped = -normalized_advantages * jnp.clip(
        ratio, 1 - clip_coef, 1 + clip_coef
    )

    return jnp.maximum(actor_loss_unclipped, actor_loss_clipped).mean()


def compute_kl_and_ratio(newlogprob, logprobs):
    """Compute approximate kl-divergence between old logprobs and new logprobs as well\
          as the ratio between them"""
    logratio = newlogprob - logprobs
    ratio = jnp.exp(logratio)

    # Calculate how much policy is changing
    approx_kl = ((ratio - 1) - logratio).mean()
    return approx_kl, ratio


def compte_PPO_critic_loss(next_values, returns, values):
    """Compute the critic loss according to PPO rule"""
    # # Value loss
    # value_loss_unclipped = (next_values - returns) ** 2
    # value_clipped = values + jnp.clip(
    #     next_values - values,
    #     clip_coef,
    #     clip_coef,
    # )
    # value_loss_unclipped = (value_loss_unclipped - returns) ** 2
    # value_loss_max = jnp.maximum(value_loss_unclipped, value_clipped)

    TD_gae_error = values - (returns + next_values)
    mse_error = (TD_gae_error**2).mean()
    # return 0.5 * value_loss_max.mean()
    return mse_error


def ppo_loss(
    agent_state: AgentState,
    agent_params: AgentParams,
    trajectory_and_variables: Tuple[Array, Array, Array, Array, Array, Array],
    clip_coef=0.1,
    vf_coef=0.5,
    ent_coef=0.1,
):  # pylint: disable=R0914
    """Compute the PPO loss given the buffer content provided and the hyperparameters"""
    obs, actions, logprobs, advantages, returns, values = trajectory_and_variables
    newlogprob, entropy, new_value = compute_logprob_entropy_value_for_update(
        agent_state, agent_params, obs, actions
    )
    approx_kl, ratio = compute_kl_and_ratio(newlogprob=newlogprob, logprobs=logprobs)
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = compute_PPO_actor_loss(normalized_advantages, ratio, clip_coef)
    value_loss = compte_PPO_critic_loss(new_value, returns, values)
    entropy_loss = entropy.mean()

    loss = actor_loss - ent_coef * entropy_loss + value_loss * vf_coef
    return loss, (actor_loss, value_loss, entropy_loss, stop_gradient(approx_kl))


class PPO:  # pylint: disable=R0902, R0913, R0914
    """PPO Agent class providing easy access to init, train and test"""

    def __init__(
        self,
        seed: int,
        num_envs: int,
        num_steps: int,
        env: SyncVectorEnv,
        actor_architecture: Sequence[Union[ActivationFunction, str]] = (
            "64",
            "relu",
            "64",
            "relu",
        ),
        critic_architecture: Sequence[Union[ActivationFunction, str]] = (
            "64",
            "relu",
            "64",
            "relu",
        ),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        clip_coef: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        logging_config: LoggingConfig = None,
    ) -> None:
        """Init"""
        self.seed = seed
        key = random.PRNGKey(seed)
        self.key, actor_key, critic_key, self.action_key, self.permutation_key = (
            random.split(key, num=5)
        )
        self.batch_size = num_envs * num_steps  # size of the batch after one rollout
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.env = env
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_coef, self.entropy_coef, self.vf_coef = (
            clip_coef,
            entropy_coef,
            vf_coef,
        )
        self.global_step = 0

        actor, critic = init_networks(env, actor_architecture, critic_architecture)

        tx = get_adam_tx(learning_rate=learning_rate)

        self.agent_state = init_agent_state(
            actor=actor,
            critic=critic,
            actor_key=actor_key,
            critic_key=critic_key,
            env=env,
            tx=tx,
        )

        self.logging = False
        if logging_config is not None:
            self.logging = True
            init_logging(logging_config)

    def rollout(
        self,
        agent_state: AgentState,
        env: Union[gym.Env, SyncVectorEnv],
        num_steps: int,
        obs: ndarray,
        done: bool,
        buffer: Buffer,
        key: random.PRNGKeyArray,
    ) -> Tuple[
        ndarray,
        Union[bool, ndarray],
        Buffer,
        random.PRNGKeyArray,
        Union[gym.Env, SyncVectorEnv],
    ]:
        """Rollout the policy on the environment for num_steps starting in obs/done"""
        for step in range(num_steps):
            self.global_step += 1 * self.num_envs
            buffer, action, key = predict_action_and_value_then_update_buffer(
                agent_state=agent_state,
                buffer=buffer,
                obs=obs,
                done=done,
                key=key,
                step=step,
            )
            obs, reward, terminated, truncated, infos = env.step(
                device_get(action.squeeze())
            )
            new_done = terminated | truncated
            buffer = insert_buffer(buffer, step, rewards=reward)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        log_variables(
                            {
                                "episodic_return": info["episode"]["r"],
                                "episodic_length": info["episode"]["l"],
                            }
                        )

        return obs, new_done, buffer, key, env

    def get_action(self, obs: ndarray) -> ndarray:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        action_logits = predict_action_logits(
            self.agent_state, self.agent_state.params, obs
        )
        self.key, subkey = random.split(self.key)
        probs = tfp.Categorical(action_logits)
        action = probs.sample(seed=subkey)
        return action

    def test(self, env: gym.Env, num_episodes: int) -> Sequence[Union[float, int]]:
        """Tests the agent's performance (sum of episodic rewards) over \
            num_episodes on env"""
        rewards = []

        for episode in range(num_episodes):
            done = False
            reward_sum = 0
            obs, _ = env.reset(seed=episode)
            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated | truncated
                reward_sum += reward
            rewards.append(reward_sum)
            log_variables(
                {"test_episode": episode, "episodic reward sum": reward_sum},
                commit=True,
            )
        return rewards

    def train(self, env: SyncVectorEnv, total_timesteps: int):
        """Train the agent on env over total_timesteps"""
        buffer = init_buffer(
            self.num_steps,
            self.num_envs,
            get_env_observation_shape(env),
            get_env_action_shape(env),
        )
        obs, _ = env.reset(seed=self.seed)
        num_updates = total_timesteps // self.batch_size
        done = jnp.zeros(self.num_envs)

        for _ in tqdm(range(1, num_updates + 1)):
            obs, done, buffer, self.action_key, env = self.rollout(
                agent_state=self.agent_state,
                env=env,
                num_steps=self.num_steps,
                obs=obs,
                done=done,
                buffer=buffer,
                key=self.action_key,
            )
            next_value = predict_value(
                self.agent_state, self.agent_state.params, obs
            ).squeeze()
            buffer_with_advantages = update_gae_advantages(
                buffer, done, next_value, self.gamma, self.gae_lambda
            )
            buffer_with_returns = update_returns(buffer_with_advantages)
            (
                self.agent_state,
                variables_to_log,
                self.permutation_key,
            ) = self.update_ppo(
                self.agent_state,
                buffer_with_returns,
                self.permutation_key,
                self.batch_size,
            )
            if self.logging:
                variables_to_log["global step"] = self.global_step
                log_variables(variables_to_log, commit=True)
        env.close()

    def update_ppo(
        self,
        agent_state: AgentState,
        buffer: Buffer,
        key: random.PRNGKeyArray,
        batch_size: int,
        update_epochs: int = 4,
        num_minibatches: int = 4,
    ) -> Tuple[AgentState, Dict[str, Array], random.PRNGKeyArray]:
        """Update PPO's agent state using the current content of the buffer and hyperparameters"""
        # Flatten collected experiences
        observation_space_shape = (buffer.obs.shape[-1],)
        action_space_shape = (buffer.actions.shape[-1],)
        obs = buffer.obs.reshape((-1,) + observation_space_shape)
        logprobs = buffer.logprobs.reshape(-1)
        actions = buffer.actions.reshape((-1,) + action_space_shape)
        advantages = buffer.advantages.reshape(-1)
        returns = buffer.returns.reshape(-1)
        values = buffer.values.reshape(-1)

        # Create function that will return gradient of the specified function
        ppo_loss_grad_function = value_and_grad(ppo_loss, argnums=1, has_aux=True)

        minibatch_size = batch_size // num_minibatches

        for _ in range(update_epochs):
            key, subkey = random.split(key)
            b_inds = random.permutation(subkey, batch_size, independent=True)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                buffer_trajectory_and_variables = (
                    obs[mb_inds],
                    actions[mb_inds],
                    logprobs[mb_inds],
                    advantages[mb_inds],
                    returns[mb_inds],
                    values[mb_inds],
                )
                (loss, (actor_loss, value_loss, entropy_loss, approx_kl)), grads = (
                    ppo_loss_grad_function(
                        agent_state,
                        agent_state.params,
                        buffer_trajectory_and_variables,
                        self.clip_coef,
                        self.vf_coef,
                        self.entropy_coef,
                    )
                )
                # Update an agent
                agent_state = agent_state.apply_gradients(grads=grads)

        # Calculate how good an approximation of the return is the value function
        y_pred, y_true = values, returns
        var_y = jnp.var(y_true)
        explained_var = jnp.nan if var_y == 0 else 1 - jnp.var(y_true - y_pred) / var_y
        variables_to_log = {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "kl-divergence": approx_kl.item(),
            "explained_var": explained_var,
        }
        return (
            agent_state,
            variables_to_log,
            key,
        )


if __name__ == "__main__":
    num_envs = 4
    num_steps = 32
    total_timesteps = int(2e4)
    train_envs = make_envs(
        "CartPole-v1",
        capture_video=False,
        num_envs=num_envs,
    )
    test_env = gym.make("CartPole-v1")

    agent = PPO(
        seed=42,
        num_envs=num_envs,
        num_steps=num_steps,
        env=train_envs,
        learning_rate=3e-4,
        logging_config=LoggingConfig(
            project_name="test jaxppo", run_name="untrained", config={"test": "test"}
        ),
    )
    rewards = agent.test(test_env, 10)
    del agent

    agent = PPO(
        seed=42,
        num_envs=num_envs,
        num_steps=num_steps,
        env=train_envs,
        learning_rate=1e-3,
        clip_coef=1.0,
        logging_config=LoggingConfig(
            project_name="test jaxppo", run_name="trained", config={"test": "test"}
        ),
    )
    agent.train(env=train_envs, total_timesteps=total_timesteps)

    rewards = agent.test(test_env, 10)
