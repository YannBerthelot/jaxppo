"""Pure-jax implementation of PPO"""
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import gymnax
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import random

from jaxppo.config import PPOConfig
from jaxppo.networks import (
    Network,
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
)
from jaxppo.networks import predict_probs as network_predict_probs
from jaxppo.utils import annealed_linear_schedule, get_parameterized_schedule
from jaxppo.wandb_logging import (
    LoggingConfig,
    finish_logging,
    init_logging,
    log_variables,
    wandb_log,
    wandb_test_log,
)
from jaxppo.wrappers import FlattenObservationWrapper, LogWrapper


class Transition(NamedTuple):
    """Store transitions inside a buffer"""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def _env_step_pre_partial(
    runner_state: tuple[
        TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray
    ],
    _: Any,
    actor_network: Network,
    critic_network: Network,
    action_key: random.PRNGKeyArray,
    step_key: random.PRNGKeyArray,
    env: Environment,
    env_params: EnvParams,
    num_envs: int,
) -> tuple[
    tuple[TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray], Transition
]:
    """
    Step the environment (over num_envs envs) and collect the transitions the buffer\
          according to the current actor, critic and env states. Then updates env state

    Args:
        runner_state (tuple[ TrainState, TrainState, EnvState, jax.Array, \
            random.PRNGKeyArray ]): Current agent state \
                (actor state, critic state, env state)
        _ (Any): Blank parameter for vectorization
        actor_network (Network): The actor network to use.
        critic_network (Network): The critic network to use.
        action_key (random.PRNGKeyArray): The random key to be used in action sampling.
        step_key (random.PRNGKeyArray): The random key to be used in env stepping.
        env (Environment): The gymnax environment to step over.
        env_params (EnvParams): The gymnax environment parameters.
        num_envs (int): The number of environments to use.

    Returns:
        tuple[ tuple[TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray],\
              Transition ]: The new runner state \
                (actor state, critic state and env state)\
                and the Transition buffer
    """
    actor_state, critic_state, env_state, last_obs, rng = runner_state

    # SELECT ACTION
    pi = actor_network.apply(actor_state.params, last_obs)
    value = critic_network.apply(critic_state.params, last_obs)
    action = pi.sample(seed=action_key)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng_step = jax.random.split(step_key, num_envs)
    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        rng_step, env_state, action, env_params
    )
    transition = Transition(done, action, value, reward, log_prob, last_obs, info)
    runner_state = (actor_state, critic_state, env_state, obsv, rng)
    return runner_state, transition


def _calculate_gae_pre_partial(
    traj_batch: Transition, last_val: jax.Array, gamma: float, gae_lambda: float
) -> tuple[jax.Array, jax.Array]:
    """
    Compute gae advantages

    Args:
        traj_batch (Transition): The transition buffer
        last_val (jax.Array): The value of the last state encoutered.
        gamma (float): The discount factor to consider.
        gae_lambda (float): The gae lambda parameter to consider.

    Returns:
        tuple[jax.Array, jax.Array]: Gae and value (carry-over values to feed for next\
              iteration)\
            and the gae for actual return
    """

    def _get_advantages(
        gae_and_next_value: tuple[jax.Array, jax.Array], transition: Transition
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """
        Iteratively compute advantages in gae style using previous gae, next value\
              and transition buffer

        Args:
            gae_and_next_value (tuple[jax.Array, jax.Array]): Previous gae and next_value
            transition (Transition): The transitions to consider

        Returns:
            tuple[tuple[jax.Array, jax.Array], jax.Array]: The updated gaes + \
                the transition's values
        """
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


def _actor_loss_fn_pre_partial(
    actor_params: FrozenDict,
    traj_batch: Transition,
    gae: jax.Array,
    actor_network: Network,
    ent_coef: float,
    clip_coef: float,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """
    Compute the PPO actor loss with the current actor network, its parameters,\
          the transition buffer, the computed gae advantages, and the hyperparameters\
              entropy coef and clipping coef.

    Args:
        actor_params (FrozenDict): The actor network's parameter/weights.
        traj_batch (Transition): A batch from the transition buffer.
        gae (jax.Array): The computed gae advantages for this batch.
        actor_network (Network): The actor network.
        ent_coef (float): The entropy coefficient in the loss
        clip_coef (float): PPO's clipping coefficient.

    Returns:
        tuple[jax.Array, tuple[jax.Array, jax.Array]]: The total PPO loss as well as\
              the sub-losses for logging
    """
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


def _critic_loss_fn_pre_partial(
    critic_params: FrozenDict,
    traj_batch: Transition,
    targets: jax.Array,
    critic_network: Network,
    clip_coef: float,
) -> jax.Array:
    """
     Compute the PPO critic loss with the current critic network , its parameters, \
        the content of the batch from the transition buffer, and the value targets.

    Args:
        critic_params (FrozenDict): The critic network's parameters/weights
        traj_batch (Transition): A batch from the transition buffer.
        targets (jax.Array): The value targets for this batch.
        critic_network (Network): The critic network.

    Returns:
        jax.Array: The critic/value loss.
    """
    # RERUN NETWORK
    value = critic_network.apply(critic_params, traj_batch.obs)

    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
        -clip_coef, clip_coef
    )
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    return value_loss


def _update_minbatch_pre_partial(
    train_state: tuple[TrainState, TrainState],
    batch_info: tuple[Transition, jax.Array, jax.Array],
    actor_network: Network,
    ent_coef: float,
    clip_coef: float,
    critic_network: Network,
    log: bool,
) -> tuple[
    tuple[TrainState, TrainState], tuple[jax.Array, jax.Array, jax.Array, jax.Array]
]:
    """
    Update the actor and critic state over a minibatch of trajectories.

    Args:
        train_state (tuple[TrainState, TrainState]): The actor and critic states.
        batch_info (tuple[Transition, jax.Array, jax.Array]): The transition buffer,\
              the advantages and the value targets of the current minibatch.
        actor_network (Network): The actor/policy network
        ent_coef (float): The entropy coefficient for the actor loss.
        clip_coef (float): PPO's clipping coefficient.
        critic_network (Network): The critic/value network.
        log (bool): Wether or not to log losses in wandb

    Returns:
        tuple[ tuple[TrainState, TrainState], \
        tuple[jax.Array, jax.Array, jax.Array, jax.Array] ]:\
              The updated actor and critic states, and the losses values for logging.
    """
    traj_batch, advantages, targets = batch_info
    actor_state, critic_state = train_state

    _actor_loss_fn = partial(
        _actor_loss_fn_pre_partial,
        actor_network=actor_network,
        ent_coef=ent_coef,
        clip_coef=clip_coef,
    )
    ppo_actor_loss_grad_function = jax.value_and_grad(_actor_loss_fn, has_aux=True)
    _critic_loss_fn = partial(
        _critic_loss_fn_pre_partial, critic_network=critic_network, clip_coef=clip_coef
    )
    ppo_critic_loss_grad_function = jax.value_and_grad(_critic_loss_fn)
    (actor_loss, (simple_actor_loss, entropy_loss)), actor_grads = (
        ppo_actor_loss_grad_function(actor_state.params, traj_batch, advantages)
    )
    critic_loss, critic_grads = ppo_critic_loss_grad_function(
        critic_state.params, traj_batch, targets
    )
    actor_state = actor_state.apply_gradients(grads=actor_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)
    losses = (actor_loss, critic_loss, simple_actor_loss, entropy_loss)
    if log:
        jax.debug.callback(
            log_variables,
            {
                "Losses/actor_loss": actor_loss,
                "Losses/critic_loss": critic_loss,
                "Losses/Simple actor loss": simple_actor_loss,
                "Losses/entropy loss": entropy_loss,
            },
        )
        print("logging")

    return (actor_state, critic_state), losses


def _update_epoch_pre_partial(
    update_state: tuple[
        TrainState, TrainState, Transition, jax.Array, jax.Array, random.PRNGKeyArray
    ],
    _: Any,
    minibatch_size: int,
    num_minibatches: int,
    actor_network: Network,
    ent_coef: float,
    clip_coef: float,
    critic_network: Network,
    num_steps: int,
    num_envs: int,
    log: bool,
) -> tuple[
    tuple[
        TrainState, TrainState, Transition, jax.Array, jax.Array, random.PRNGKeyArray
    ],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """
    Update the actor and critic states over an epoch of num_minibatches minibatches.\
          Collect losses for logging.

    Args:
        update_state (tuple[ TrainState, TrainState, Transition, jax.Array, jax.Array,\
        random.PRNGKeyArray ]): \
            The states of the actor and critic networks, the transition buffer for\
             this epoch, the advantages and target values for this epoch and the\
                  permutation random key.
        _ (Any): Unused parameter, just here for having an axis for vectorizing.
        minibatch_size (int): _description_
        num_minibatches (int): _description_
        actor_network (Network): _description_
        ent_coef (float): _description_
        clip_coef (float): _description_
        critic_network (Network): _description_
        log (bool): Wether or not to log the losses in wandb

    Returns:
        tuple[ tuple[ TrainState, TrainState, Transition, jax.Array, jax.Array,/
        random.PRNGKeyArray ], tuple[jax.Array, jax.Array, jax.Array, jax.Array], ]:\
            The new update state (The states of the actor and critic networks, the\
             transition buffer for this epoch, the advantages and target values for\
             this epoch and the permutation random key) and losses for logging.
    """
    actor_state, critic_state, traj_batch, advantages, targets, rng = update_state
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
    _update_minibatch = partial(
        _update_minbatch_pre_partial,
        actor_network=actor_network,
        ent_coef=ent_coef,
        clip_coef=clip_coef,
        critic_network=critic_network,
        log=log,
    )
    (actor_state, critic_state), losses = jax.lax.scan(
        _update_minibatch, train_state, minibatches
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


def _update_step_pre_partial(  # pylint: disable=R0913,R0914
    runner_state: tuple[
        TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray
    ],
    _: Any,
    actor_network: Network,
    critic_network: Network,
    action_key: random.PRNGKeyArray,
    step_key: random.PRNGKeyArray,
    env: Environment,
    env_params: EnvParams,
    gamma: float,
    gae_lambda: float,
    permutation_key: random.PRNGKeyArray,
    minibatch_size: int,
    num_minibatches: int,
    ent_coef: float,
    clip_coef: float,
    update_epochs: int,
    num_envs: int,
    num_steps: int,
    log: bool,
) -> tuple[
    tuple[TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray], dict
]:
    """
    Update the agent (actor and critic network states) on the provided environment(s)\
          for the given number of epochs, num of minibatches and minibatch size.

    Args:
        runner_state (tuple[ TrainState, TrainState, EnvState, jax.Array,\
              random.PRNGKeyArray ]): The runner state (containing, Actor, Critic\
                  and Env states, your last observation and the new random key).
        _ (Any): Unused parameter, just here for having an axis for vectorizing.
        actor_network (Network): The actor network.
        critic_network (Network): The critic network.
        action_key (random.PRNGKeyArray): The random key for action sampling.
        step_key (random.PRNGKeyArray): The random key for step sampling.
        env (Environment): The gymnax environment to train on
        env_params (EnvParams): The gymnax environment parameters.
        gamma (float): The discount factor.
        gae_lambda (float): The gae lambda parameter for advantage computation.
        permutation_key (random.PRNGKeyArray): The random key for batch sampling.
        minibatch_size (int): The minibatch size (computed as \
            num_envs * num_steps // num_minibatches).
        num_minibatches (int): The number of minibatches in each epoch \
            (with permuted trajectories).
        ent_coef (float): The entropy coefficient for actor loss.
        clip_coef (float): PPO's clipping coefficient.
        update_epochs (int): The number of training epochs over the same experience.
        num_envs (int): The number of environments to train on in parallel.

    Returns:
       tuple[
        tuple[TrainState, TrainState, EnvState, jax.Array, \
        random.PRNGKeyArray], dict]:\
            The new runner state (containing, Actor, Critic\
                  and Env states, your last observation and the new random key).
            and a metric dict.
    """
    _env_step = partial(
        _env_step_pre_partial,
        actor_network=actor_network,
        critic_network=critic_network,
        action_key=action_key,
        step_key=step_key,
        env=env,
        env_params=env_params,
        num_envs=num_envs,
    )
    # COLLECT BATCH TRAJECTORIES
    runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, num_steps)

    # CALCULATE ADVANTAGE
    actor_state, critic_state, env_state, last_obs, rng = runner_state
    last_val = critic_network.apply(critic_state.params, last_obs)

    _calculate_gae = partial(
        _calculate_gae_pre_partial, gamma=gamma, gae_lambda=gae_lambda
    )
    advantages, targets = _calculate_gae(traj_batch, last_val)

    # UPDATE NETWORK

    update_state = (
        actor_state,
        critic_state,
        traj_batch,
        advantages,
        targets,
        permutation_key,
    )
    if log:
        jax.debug.callback(wandb_log, traj_batch.info)

    _update_epoch = partial(
        _update_epoch_pre_partial,
        minibatch_size=minibatch_size,
        num_minibatches=num_minibatches,
        actor_network=actor_network,
        ent_coef=ent_coef,
        clip_coef=clip_coef,
        critic_network=critic_network,
        num_steps=num_steps,
        num_envs=num_envs,
        log=log,
    )
    update_state, _ = jax.lax.scan(_update_epoch, update_state, None, update_epochs)  #
    actor_state = update_state[0]
    critic_state = update_state[1]
    traj_batch = update_state[2]
    metric = traj_batch.info
    rng = update_state[-1]
    runner_state = (actor_state, critic_state, env_state, last_obs, rng)
    return runner_state, metric


def make_train(  # pylint: disable=W0102, R0913
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
    log: bool = False,
    env_params: Optional[EnvParams] = None,
) -> Callable[
    ..., tuple[TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray]
]:
    """
    Generate the train function (to be jitted) according to the given parameters.

    Args:
        total_timesteps (int): Total number of timesteps (distributed across all envs)\
              to train for.
        num_steps (int): Number of steps to run per environment before updating.
        num_envs (int): Number of environments to run in parrallel.
        env_id (str): The gym-id of the environment to use.
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

    Returns:
        Callable[ ..., tuple[TrainState, TrainState, EnvState, jax.Array,\
              random.PRNGKeyArray] ]: The train function to be called to actually train.
    """

    num_updates = total_timesteps // num_steps // num_envs
    minibatch_size = num_envs * num_steps // num_minibatches
    batch_size = num_envs * num_steps
    if isinstance(env_id, str):
        env, env_params = gymnax.make(env_id)

    elif isinstance(env_id, Environment) or issubclass(env_id.__class__, GymnaxWrapper):
        if env_params is None:
            raise ValueError(
                "env_params should be provided if using a predefined env and not env_id"
            )
        env = env_id
    else:
        raise ValueError(
            f"Environment is not a gymnax env or wrapped gymnax env : {env_id}"
        )
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(
        key: random.PRNGKeyArray,
    ) -> tuple[TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray]:
        """Train the agent with the given random key and according to config"""
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

        _, _rng = jax.random.split(key)
        runner_state = (
            actor_state,
            critic_state,
            env_state,
            obsv,
            _rng,
        )
        _update_step = partial(
            _update_step_pre_partial,
            actor_network=actor_network,
            critic_network=critic_network,
            action_key=action_key,
            step_key=step_key,
            env=env,
            env_params=env_params,
            gamma=gamma,
            gae_lambda=gae_lambda,
            permutation_key=permutation_key,
            minibatch_size=minibatch_size,
            num_minibatches=num_minibatches,
            ent_coef=ent_coef,
            clip_coef=clip_coef,
            update_epochs=update_epochs,
            num_envs=num_envs,
            num_steps=num_steps,
            log=log,
        )
        runner_state, _ = jax.lax.scan(_update_step, runner_state, None, num_updates)

        return runner_state

    return train


class PPO:
    """PPO Agent that allows simple training and testing"""

    def __init__(  # pylint: disable=W0102, R0913
        self,
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
        log=False,
        env_params: Optional[EnvParams] = None,
    ) -> None:
        """
        PPO Agent that allows simple training and testing

        Args:
        total_timesteps (int): Total number of timesteps (distributed across all envs)\
              to train for.
        num_steps (int): Number of steps to run per environment before updating.
        num_envs (int): Number of environments to run in parrallel.
        env_id (str): The gym-id of the environment to use.
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
            ent_coef=ent_coef,
            log=log,
            env_params=env_params,
        )

        self._actor_state = None
        self._critic_state = None
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

        if self.config.log:
            logging_config = LoggingConfig(
                project_name="test pure jax ppo",
                run_name=f"{seed=}",
                config=self.config.model_dump(mode="json"),
            )
            init_logging(logging_config)

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
            ent_coef=self.config.ent_coef,
            log=self.config.log,
            env_params=self.config.env_params,
        )
        runner_state = train_jit(key)
        self._actor_state, self._critic_state, _, _, _ = runner_state
        if test:
            self.test(self.config.num_episode_test, seed=seed)
        if self.config.log:
            finish_logging()

    def predict_probs(self, obs: jax.Array) -> jax.Array:
        """Predict the policy's probabilities of taking actions in obs"""
        if self._actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )
        return network_predict_probs(self._actor_state, self._actor_state.params, obs)

    def get_action(self, obs: jax.Array, key: random.PRNGKeyArray) -> jax.Array:
        """Returns a numpy action compliant with gym using the current \
            state of the agent"""
        if self._actor_state is None:
            raise ValueError(
                "Attempted to predict probs without an actor state/training the agent"
                " first"
            )
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
        env, env_params = gymnax.make(self.config.env_id)

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
        if self.config.log:
            wandb_test_log(rewards)


if __name__ == "__main__":
    num_envs = 8
    total_timesteps = int(1e6)
    num_steps = 128
    learning_rate = 2.5e-4
    clip_coef = 0.2
    entropy_coef = 0.01
    env_id = "CartPole-v1"

    agent = PPO(
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        num_envs=num_envs,
        env_id=env_id,
        learning_rate=learning_rate,
        actor_architecture=["64", "tanh", "64", "tanh"],
        critic_architecture=["64", "tanh", "64", "tanh"],
        log=True,
    )
    agent.train(seed=42, test=True)
