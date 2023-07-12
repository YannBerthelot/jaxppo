"""Implementation of PPO using predefined Networks and Buffer"""
from typing import Dict, Sequence, Tuple, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfp
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnax import EnvParams
from gymnax.environments.environment import Environment, EnvState
from jax import Array, device_get, jit, random, value_and_grad
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
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
    predict_action_logits,
    predict_value,
)
from jaxppo.utils import (
    annealed_linear_schedule,
    get_env_action_shape,
    get_env_observation_shape,
    get_parameterized_schedule,
    make_gymnax_env,
)
from jaxppo.wandb_logging import (
    LoggingConfig,
    finish_logging,
    init_logging,
    log_variables,
)


@jit
def get_logprob_and_action(
    key: random.PRNGKeyArray, logits: Array
) -> Tuple[Array, Array, random.PRNGKeyArray]:
    """Create a distribution from the logits, sample an action and compute its logprob"""
    key, subkey = random.split(key)
    probs = tfp.Categorical(logits)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action)
    return logprob, action, key


@jit
def predict_action_and_value_then_update_buffer(
    actor_state: TrainState,
    critic_state: TrainState,
    buffer: Buffer,
    obs: ndarray,
    key: random.PRNGKeyArray,
    step: int,
) -> Tuple[Buffer, Array, random.PRNGKeyArray]:
    """
    -Select an action using the current actor-network state
    -Compute the value of the current obs
    -Compute log-prob of the selected action
    -Store these values into the buffer
    """
    value = predict_value(critic_state, critic_state.params, obs)
    action_logits = predict_action_logits(actor_state, actor_state.params, obs)
    logprob, action, key = get_logprob_and_action(key, action_logits)
    buffer = insert_buffer(
        buffer,
        step,
        obs=obs,
        actions=action,
        logprobs=logprob,
        values=value.squeeze(),
    )
    return buffer, action, key


# @jit
# def compute_logprob_entropy_value_for_update(agent_state, agent_params, obs, action):
#     """Compute values needed for the PPO loss based on the agent state and \
#         observed action and obs"""
#     action_logits = predict_action_logits(agent_state, agent_params, obs)
#     value = predict_value(agent_state, agent_params, obs)
#     probs = tfp.Categorical(action_logits)
#     return probs.log_prob(action), probs.entropy(), value.squeeze()


@jit
def compute_logprob_entropy_for_update(
    actor_state: TrainState, actor_params: FrozenDict, obs: Array, action: Array
):
    """Compute values needed for the PPO loss based on the agent state and \
        observed action and obs"""
    action_logits = predict_action_logits(actor_state, actor_params, obs)
    probs = tfp.Categorical(action_logits)
    return probs.log_prob(action), probs.entropy()


@jit
def compute_value_for_update(
    critic_state: TrainState, critic_params: FrozenDict, obs: Array
) -> Array:
    """Compute value based on critic state, critic params and observation."""
    return predict_value(critic_state, critic_params, obs)


@jit
def compute_PPO_actor_loss(
    normalized_advantages: Array, ratio: float, clip_coef: float
) -> Array:
    """Compute the actor loss according to PPO rule"""
    # Policy loss
    actor_loss_unclipped = -normalized_advantages * ratio
    actor_loss_clipped = -normalized_advantages * jnp.clip(
        ratio, 1 - clip_coef, 1 + clip_coef
    )
    return jnp.maximum(actor_loss_unclipped, actor_loss_clipped).mean()


@jit
def compute_kl_and_ratio(newlogprob, logprobs):
    """Compute approximate kl-divergence between old logprobs and new logprobs as well\
          as the ratio between them"""
    logratio = newlogprob - logprobs
    ratio = jnp.exp(logratio)

    # Calculate how much policy is changing
    approx_kl = ((ratio - 1) - logratio).mean()
    return approx_kl, ratio


@jit
def compte_PPO_critic_loss(next_values, returns):
    """Compute the critic loss according to PPO rule"""
    return 0.5 * ((next_values - returns) ** 2).mean()


@jit
def ppo_actor_loss(
    actor_state, actor_params, trajectory_and_variables, clip_coef, ent_coef=0.01
):  # pylint: disable=R0914
    """Compute the actor loss according to PPO rules"""
    obs, actions, logprobs, advantages, _, _ = trajectory_and_variables
    newlogprob, entropy = compute_logprob_entropy_for_update(
        actor_state, actor_params, obs, actions
    )

    # if not jnp.array_equal(advantages - advantages.mean(), jnp.zeros(advantages.shape)):
    #     normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # else:
    #     normalized_advantages = advantages
    normalized_advantages = jax.lax.cond(
        jnp.array_equal(advantages - advantages.mean(), jnp.zeros(advantages.shape)),
        lambda: advantages,
        lambda: (advantages - advantages.mean()) / (advantages.std() + 1e-8),
    )
    # normalized_advantages = advantages
    assert (
        newlogprob.shape == logprobs.shape
    ), f"{newlogprob.shape=}, {logprobs.shape=} "
    approx_kl, ratio = compute_kl_and_ratio(newlogprob=newlogprob, logprobs=logprobs)
    simple_actor_loss = compute_PPO_actor_loss(normalized_advantages, ratio, clip_coef)
    entropy_loss = entropy.mean()
    actor_loss = simple_actor_loss + ent_coef * entropy_loss
    return actor_loss, (
        simple_actor_loss,
        entropy_loss,
        stop_gradient(approx_kl),
        stop_gradient(advantages),
        stop_gradient(normalized_advantages),
    )


@jit
def ppo_critic_loss(
    critic_state,
    critic_params,
    trajectory_and_variables,
):
    """Compute the critic loss according to PPO rules"""
    obs, _, _, _, returns, _ = trajectory_and_variables
    new_value = compute_value_for_update(critic_state, critic_params, obs)
    return compte_PPO_critic_loss(new_value, returns)


def get_flattened_experience(buffer, action_space_shape, observation_space_shape):
    """Extract and reshape to flattened experience from the buffer"""
    # observation_space_shape = (buffer.obs.shape[-1],)
    # action_space_shape = (buffer.actions.shape[-1],)
    obs = buffer.obs.reshape((-1,) + observation_space_shape)
    logprobs = buffer.logprobs.reshape(-1)
    actions = buffer.actions.reshape((-1,) + action_space_shape)
    advantages = buffer.advantages.reshape(-1)
    returns = buffer.returns.reshape(-1)
    values = buffer.values.reshape(-1)
    return obs, logprobs, actions, advantages, returns, values


@jit
def compute_explained_var(values, returns):
    """Compute explained variance based on values and returns"""
    # Calculate how good an approximation of the return is the value function
    y_pred, y_true = values, returns
    var_y = jnp.var(y_true)
    return jax.lax.cond(
        var_y == 0, lambda: jnp.nan, lambda: 1 - jnp.var(y_true - y_pred) / var_y
    )
    # return jnp.nan if var_y == 0 else 1 - jnp.var(y_true - y_pred) / var_y


class PPO:  # pylint: disable=R0902, R0913, R0914
    """PPO Agent class providing easy access to init, train and test"""

    def __init__(
        self,
        seed: int,
        num_envs: int,
        num_steps: int,
        env: Environment,
        env_params: EnvParams,
        actor_architecture: Sequence[Union[ActivationFunction, str]] = (
            "64",
            "tanh",
            "64",
            "tanh",
        ),
        critic_architecture: Sequence[Union[ActivationFunction, str]] = (
            "64",
            "tanh",
            "64",
            "tanh",
        ),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        clip_coef: float = 0.2,
        entropy_coef: float = 0.01,
        logging_config: LoggingConfig = None,
        num_minibatches=4,
        update_epochs=4,
        total_timesteps=5e4,
    ) -> None:
        """Init"""
        self.seed = seed
        key = random.PRNGKey(seed)
        (
            self.key,
            actor_key,
            critic_key,
            self.action_key,
            self.permutation_key,
            self.reset_key,
        ) = random.split(key, num=6)
        self.batch_size = num_envs * num_steps  # size of the batch after one rollout
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.env = env
        self.env_params = env_params
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_coef, self.entropy_coef = (
            clip_coef,
            entropy_coef,
        )
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.global_step = 0

        actor_network, critic_network = init_networks(
            self.env, actor_architecture, critic_architecture
        )
        batch_size = int(num_envs * num_steps)
        scheduler = get_parameterized_schedule(
            linear_scheduler=annealed_linear_schedule,
            initial_learning_rate=learning_rate,
            num_minibatches=num_minibatches,
            update_epochs=update_epochs,
            num_updates=total_timesteps // batch_size,
        )
        tx = get_adam_tx(learning_rate=scheduler)

        self.actor_state, self.critic_state = init_actor_and_critic_state(
            actor_network=actor_network,
            critic_network=critic_network,
            actor_key=actor_key,
            critic_key=critic_key,
            env=env,
            tx=tx,
            env_params=env_params,
        )

        self.logging = False
        if logging_config is not None:
            self.logging = True
            init_logging(logging_config)

    def rollout(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        env: Union[gym.Env, SyncVectorEnv],
        env_state: EnvState,
        num_steps: int,
        obs: ndarray,
        done: bool,
        buffer: Buffer,
        key: random.PRNGKeyArray,
        step_key: random.PRNGKeyArray,
    ) -> tuple[
        ndarray,
        Union[bool, ndarray],
        Buffer,
        random.PRNGKeyArray,
        Union[gym.Env, SyncVectorEnv, Environment],
        EnvState,
    ]:
        """Rollout the policy on the environment for num_steps starting in obs/done"""
        for step in range(num_steps):
            self.global_step += 1 * self.num_envs
            key, predict_key = random.split(key, num=2)
            buffer, action, key = predict_action_and_value_then_update_buffer(
                actor_state=actor_state,
                critic_state=critic_state,
                buffer=buffer,
                obs=obs,
                key=predict_key,
                step=step,
            )
            buffer = insert_buffer(buffer, step, dones=done)

            # gym
            vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
            obs, env_state, reward, done, infos = vmap_step(
                step_key, env_state, device_get(action.squeeze()), self.env_params
            )
            # gym
            buffer = insert_buffer(buffer, step, rewards=reward)
            self.log_final_info(infos)

        return obs, done, buffer, key, env, env_state

    def log_final_info(self, infos):
        """Log the episode returns at the end of episodes using the infos dict"""
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    if self.logging and "episode" in info:
                        log_variables(
                            {
                                "episodic_return": info["episode"]["r"],
                                "episodic_length": info["episode"]["l"],
                            }
                        )

    def get_action(self, obs: ndarray) -> ndarray:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        action_logits = predict_action_logits(
            self.actor_state, self.actor_state.params, obs
        )
        self.key, subkey = random.split(self.key)
        probs = tfp.Categorical(action_logits)
        action = probs.sample(seed=subkey)
        return action

    def get_probs(self, obs: ndarray) -> ndarray:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        action_logits = predict_action_logits(
            self.actor_state, self.actor_state.params, obs
        )
        probs = tfp.Categorical(action_logits)
        return probs.probs_parameter()

    def get_value(self, obs: ndarray) -> ndarray:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        return predict_value(self.critic_state, self.critic_state.params, obs)

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
            if self.logging:
                log_variables(
                    {"test_episode": episode, "episodic reward sum": reward_sum},
                    commit=True,
                )
        return rewards

    def train(self, env: Environment, total_timesteps: int):
        """Train the agent on env over total_timesteps"""
        buffer = init_buffer(
            self.num_steps,
            self.num_envs,
            get_env_observation_shape(env, self.env_params),
            get_env_action_shape(env),
        )
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_keys = jax.random.split(self.reset_key, self.num_envs)
        obs, env_state = vmap_reset(vmap_keys, self.env_params)
        num_updates = total_timesteps // self.batch_size
        done = jnp.zeros(self.num_envs)
        for _ in tqdm(range(1, num_updates + 1)):
            new_obs, done, buffer, self.action_key, env, env_state = self.rollout(
                actor_state=self.actor_state,
                critic_state=self.critic_state,
                env=env,
                env_state=env_state,
                num_steps=self.num_steps,
                obs=obs,
                done=done,
                buffer=buffer,
                key=self.action_key,
                step_key=vmap_keys,
            )
            obs = new_obs
            next_value = predict_value(
                self.critic_state, self.critic_state.params, new_obs
            ).squeeze()
            buffer_with_advantages = update_gae_advantages(
                buffer, done, next_value, self.gamma, self.gae_lambda
            )
            buffer_with_returns = update_returns(buffer_with_advantages)
            (
                self.actor_state,
                self.critic_state,
                variables_to_log,
                self.permutation_key,
            ) = self.update_ppo(
                self.actor_state,
                self.critic_state,
                buffer_with_returns,
                self.permutation_key,
                self.batch_size,
            )
            if self.logging:
                variables_to_log["global step"] = self.global_step
                log_variables(variables_to_log, commit=True)

    #  env.close()

    def update_ppo(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        buffer: Buffer,
        key: random.PRNGKeyArray,
        batch_size: int,
        update_epochs: int = 4,
        num_minibatches: int = 4,
    ) -> Tuple[TrainState, TrainState, Dict[str, Array], random.PRNGKeyArray]:
        """Update PPO's agent state using the current content of the buffer and hyperparameters"""
        # Flatten collected experiences
        obs, logprobs, actions, advantages, returns, values = get_flattened_experience(
            buffer,
            get_env_action_shape(self.env),
            get_env_observation_shape(self.env, self.env_params),
        )
        # Create function that will return gradient of the specified function
        # ppo_loss_grad_function = jit(value_and_grad(ppo_loss, argnums=1, has_aux=True))

        ppo_actor_loss_grad_function = jit(
            value_and_grad(ppo_actor_loss, argnums=1, has_aux=True)
        )
        ppo_critic_loss_grad_function = jit(
            value_and_grad(ppo_critic_loss, argnums=1, has_aux=False)
        )
        # ppo_loss_grad_function = jit(value_and_grad(ppo_loss, argnums=1, has_aux=True))

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
                actor_loss = ppo_actor_loss(
                    actor_state,
                    actor_state.params,
                    buffer_trajectory_and_variables,
                    self.clip_coef,
                    self.entropy_coef,
                )

                (
                    actor_loss,
                    (
                        simple_actor_loss,
                        entropy_loss,
                        approx_kl,
                        log_advantages,
                        log_normalized_advantages,
                    ),
                ), actor_grads = ppo_actor_loss_grad_function(
                    actor_state,
                    actor_state.params,
                    buffer_trajectory_and_variables,
                    self.clip_coef,
                    self.entropy_coef,
                )
                critic_loss, critic_grads = ppo_critic_loss_grad_function(
                    critic_state,
                    critic_state.params,
                    buffer_trajectory_and_variables,
                )
                # Update an agent
                actor_state = actor_state.apply_gradients(grads=actor_grads)
                critic_state = critic_state.apply_gradients(grads=critic_grads)

        explained_var = compute_explained_var(values, returns)

        variables_to_log = {
            "actor_loss": actor_loss.item(),
            "simple_actor_loss": simple_actor_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "kl-divergence": approx_kl.item(),
            "explained_var": explained_var,
            "advantages": log_advantages,
            "normalized_advantages": log_normalized_advantages,
            "returns": buffer.returns,
        }
        return (
            actor_state,
            critic_state,
            variables_to_log,
            key,
        )


if __name__ == "__main__":
    num_envs = 4
    num_steps = 32
    total_timesteps = int(5e4)
    train_env, train_env_params, keys = make_gymnax_env("CartPole-v1", seed=42)
    test_env = gym.make("CartPole-v1")
    agent = PPO(
        seed=42,
        num_envs=num_envs,
        num_steps=128,
        env=train_env,
        env_params=train_env_params,
        learning_rate=1e-2,
        clip_coef=0.2,
        entropy_coef=0.00,
        logging_config=LoggingConfig(
            project_name="test jaxppo", run_name="trained", config={"test": "test"}
        ),
        total_timesteps=total_timesteps,
    )
    agent.train(env=train_env, total_timesteps=total_timesteps)
    rewards = agent.test(test_env, 10)
    finish_logging()
