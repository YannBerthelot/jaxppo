import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from jaxppo.networks import init_actor_and_critic_state, init_networks, get_adam_tx
from jaxppo.utils import get_parameterized_schedule, annealed_linear_schedule
from jaxppo.wrappers import FlattenObservationWrapper, LogWrapper
import wandb
from jaxppo.wandb_logging import (
    LoggingConfig,
    finish_logging,
    init_logging,
    log_variables,
)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(
    total_timesteps: int,
    num_steps: int,
    num_envs: int,
    env_id: str,
    learning_rate: float,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    actor_architecture=["64", "tanh", "64", "tanh"],
    critic_architecture=["64", "relu", "relu", "tanh"],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    debug: bool = False,
):
    num_updates = total_timesteps // num_steps // num_envs
    minibatch_size = num_envs * num_steps // num_minibatches
    batch_size = num_envs * num_steps
    env, env_params = gymnax.make(env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env)

    def train(key):
        # INIT NETWORK

        (
            key,
            actor_key,
            critic_key,
            action_key,
            permutation_key,
            reset_key,
            step_key,
        ) = random.split(key, num=7)
        actor_network, critic_network = init_networks(
            env=env,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            squeeze_value=True,
            categorical_output=True,
        )
        scheduler = get_parameterized_schedule(
            linear_scheduler=annealed_linear_schedule,
            initial_learning_rate=learning_rate,
            num_minibatches=num_minibatches,
            update_epochs=update_epochs,
            num_updates=total_timesteps // batch_size,
        )
        tx = get_adam_tx(learning_rate=scheduler)
        actor_state, critic_state = init_actor_and_critic_state(
            actor_network=actor_network,
            critic_network=critic_network,
            actor_key=actor_key,
            critic_key=critic_key,
            env=env,
            tx=tx,
            env_params=env_params,
        )

        # INIT ENV
        reset_rng = jax.random.split(reset_key, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                actor_state, critic_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                pi = actor_network.apply(actor_state.params, last_obs)
                value = critic_network.apply(critic_state.params, last_obs)
                action = pi.sample(seed=action_key)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng_step = jax.random.split(step_key, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (actor_state, critic_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, num_steps
            )

            # CALCULATE ADVANTAGE
            actor_state, critic_state, env_state, last_obs, rng = runner_state
            last_val = critic_network.apply(critic_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = delta + gamma * gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    actor_state, critic_state = train_state

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        # RERUN NETWORK
                        pi = actor_network.apply(actor_params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - clip_coef,
                                1.0 + clip_coef,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor - ent_coef * entropy
                        return total_loss, (loss_actor, entropy)

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        # RERUN NETWORK
                        value = critic_network.apply(critic_params, traj_batch.obs)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-clip_coef, clip_coef)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        return value_loss

                    ppo_actor_loss_grad_function = jax.value_and_grad(
                        _actor_loss_fn, has_aux=True
                    )
                    ppo_critic_loss_grad_function = jax.value_and_grad(_critic_loss_fn)
                    (actor_loss, (simple_actor_loss, entropy_loss)), actor_grads = (
                        ppo_actor_loss_grad_function(
                            actor_state.params, traj_batch, advantages
                        )
                    )
                    critic_loss, critic_grads = ppo_critic_loss_grad_function(
                        critic_state.params, traj_batch, targets
                    )
                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    critic_state = critic_state.apply_gradients(grads=critic_grads)
                    losses = (actor_loss, critic_loss, simple_actor_loss, entropy_loss)
                    return (actor_state, critic_state), losses

                actor_state, critic_state, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)
                batch_size = minibatch_size * num_minibatches
                assert (
                    batch_size == num_steps * num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                train_state = (actor_state, critic_state)
                (actor_state, critic_state), losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    actor_state,
                    critic_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )

                return update_state, losses

            update_state = (
                actor_state,
                critic_state,
                traj_batch,
                advantages,
                targets,
                permutation_key,
            )

            def wandb_log(info):
                return_values = info["returned_episode_returns"][
                    info["returned_episode"]
                ]
                wandb.log({"mean_returns_over_batch": jnp.mean(return_values)})

            jax.debug.callback(wandb_log, traj_batch.info)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, update_epochs
            )
            actor_state = update_state[0]
            critic_state = update_state[1]
            traj_batch = update_state[2]
            metric = traj_batch.info
            rng = update_state[-1]
            if debug:

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = info["timestep"][info["returned_episode"]] * num_envs
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic"
                            f" return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (actor_state, critic_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(key)
        runner_state = (actor_state, critic_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    for seed in range(1, 11):
        key = random.PRNGKey(seed)
        logging_config = LoggingConfig(
            project_name="test pure jax ppo",
            run_name=f"{seed=}",
            config={"test": "test"},
        )
        init_logging(logging_config)
        num_envs = 4
        total_timesteps = int(1e6)
        num_steps = 128
        learning_rate = 2.5e-4
        clip_coef = 0.2
        entropy_coef = 0.01
        env_id = "CartPole-v1"

        train_jit = jax.jit(
            make_train(
                total_timesteps=total_timesteps,
                num_steps=num_steps,
                num_envs=num_envs,
                env_id=env_id,
                learning_rate=learning_rate,
                debug=False,
                actor_architecture=["64", "tanh", "64", "tanh"],
                critic_architecture=["64", "tanh", "64", "tanh  "],
            )
        )
        out = train_jit(key)
        finish_logging()
