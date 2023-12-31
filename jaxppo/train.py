"""Pure jax PPO training"""
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

from jaxppo.networks import (
    Network,
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
)
from jaxppo.utils import annealed_linear_schedule, get_parameterized_schedule
from jaxppo.wandb_logging import wandb_log
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
    env: Environment,
    env_params: EnvParams,
    num_envs: int,
    shared_network: bool = False,
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
    if shared_network:
        pi, value = actor_state.apply_fn(actor_state.params, last_obs)
    else:
        pi = actor_state.apply_fn(actor_state.params, last_obs)
        value = critic_state.apply_fn(critic_state.params, last_obs)
    rng, action_key = jax.random.split(rng)
    action = pi.sample(seed=action_key)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, step_key = jax.random.split(rng)
    rng_step = jax.random.split(step_key, num_envs)
    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        rng_step, env_state, action, env_params
    )
    transition = Transition(done, action, value, reward, log_prob, last_obs, info)
    runner_state = (actor_state, critic_state, env_state, obsv, rng)
    return runner_state, transition


# @partial(jax.jit, static_argnums=[2, 3])
def _calculate_gae(
    traj_batch: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
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
        # current estimation of gae + value at t+1 because we have working in reverse
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )

        next_state_is_non_terminal = 1.0 - done

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * next_state_is_non_terminal * gae

        # tuple is carry-over state for scan, gae after the comma is the actual return at the end of the scan
        return (
            gae,
            value,
        ), gae

    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=(jnp.zeros_like(last_val), last_val),
        xs=traj_batch,
        reverse=True,
        unroll=16,
    )
    returns = advantages + traj_batch.value
    return advantages, returns


def _actor_loss_fn_pre_partial(
    actor_params: FrozenDict,
    traj_batch: Transition,
    gae: jax.Array,
    actor_network: Network,
    ent_coef: float,
    clip_coef: float,
    advantage_normalization: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
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
    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = -ratio * gae
    loss_actor2 = (
        -jnp.clip(
            ratio,
            1.0 - clip_coef,
            1.0 + clip_coef,
        )
        * gae
    )
    loss_actor = jnp.maximum(loss_actor1, loss_actor2).mean()

    # CALCULATE AUXILIARIES
    clip_fraction = (jnp.abs(ratio - 1) > clip_coef).mean()
    entropy = pi.entropy().mean()

    total_loss = loss_actor - ent_coef * entropy
    auxiliaries = (loss_actor, entropy, jax.lax.stop_gradient(clip_fraction))
    return total_loss, auxiliaries


def _critic_loss_fn_pre_partial(
    critic_params: FrozenDict,
    traj_batch: Transition,
    targets: jax.Array,
    critic_network: Network,
    clip_coef_vf: Optional[float],
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
    value_losses = (value - targets) ** 2

    # CALCULATE VALUE LOSS
    if clip_coef_vf is not None:
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -clip_coef_vf, clip_coef_vf
        )
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(), (
            targets.mean(),
            value_pred_clipped.mean(),
        )
    value_loss = 0.5 * value_losses.mean()
    auxiliaries = (
        jax.lax.stop_gradient(targets).mean(),
        jax.lax.stop_gradient(value).mean(),
    )
    return value_loss, auxiliaries


def _loss_fn_pre_partial(
    network_params: FrozenDict,
    traj_batch: Transition,
    targets: jax.Array,
    gae: jax.Array,
    network: Network,
    clip_coef: float,
    clip_coef_vf: Optional[float],
    vf_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
):
    # RERUN NETWORK
    pi, value = network.apply(network_params, traj_batch.obs)

    # CALCULATE ACTOR LOSS
    log_prob = pi.log_prob(traj_batch.action)
    ratio = jnp.exp(log_prob - traj_batch.log_prob)

    if advantage_normalization:
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
    policy_loss = loss_actor - ent_coef * entropy

    # CALCULATE CRITIC LOSS
    value_losses = (targets - value) ** 2
    if clip_coef_vf is not None:
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -clip_coef_vf, clip_coef_vf
        )
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = value_losses.mean()

    # CALCULATE COMBINED LOSS
    total_loss = vf_coef * value_loss + policy_loss
    clip_fraction = (jnp.abs(ratio - 1) > clip_coef).mean()
    auxiliaries = (
        policy_loss,
        value_loss,
        loss_actor,
        entropy,
        clip_fraction,
        targets,
        value,
    )
    return total_loss, auxiliaries


def _update_minbatch_pre_partial(  # pylint: disable=R0914
    train_state: tuple[TrainState, TrainState],
    batch_info: tuple[Transition, jax.Array, jax.Array],
    actor_network: Network,
    ent_coef: float,
    clip_coef: float,
    clip_coef_vf: Optional[float],
    critic_network: Network,
    shared_network: bool,
    vf_coef: float,
    advantage_normalization: bool,
) -> tuple[tuple[TrainState, TrainState], tuple[jax.Array, ...],]:
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

    if shared_network:
        _loss_fn = partial(
            _loss_fn_pre_partial,
            network=actor_network,
            ent_coef=ent_coef,
            clip_coef=clip_coef,
            clip_coef_vf=clip_coef_vf,
            vf_coef=vf_coef,
            advantage_normalization=advantage_normalization,
        )
        ppo_loss_grad_function = jax.value_and_grad(_loss_fn, has_aux=True)
        (
            total_loss,
            (
                actor_loss,
                critic_loss,
                simple_actor_loss,
                entropy_loss,
                clip_fraction,
                update_targets,
                values,
            ),
        ), grads = ppo_loss_grad_function(
            actor_state.params, traj_batch, targets, advantages
        )
        actor_state = actor_state.apply_gradients(grads=grads)
    else:
        _actor_loss_fn = partial(
            _actor_loss_fn_pre_partial,
            actor_network=actor_network,
            ent_coef=ent_coef,
            clip_coef=clip_coef,
            advantage_normalization=advantage_normalization,
        )
        ppo_actor_loss_grad_function = jax.value_and_grad(_actor_loss_fn, has_aux=True)
        _critic_loss_fn = partial(
            _critic_loss_fn_pre_partial,
            critic_network=critic_network,
            clip_coef_vf=clip_coef_vf,
        )
        ppo_critic_loss_grad_function = jax.value_and_grad(
            _critic_loss_fn, has_aux=True
        )
        (actor_loss, (simple_actor_loss, entropy_loss, clip_fraction)), actor_grads = (
            ppo_actor_loss_grad_function(actor_state.params, traj_batch, advantages)
        )
        (critic_loss, (update_targets, values)), critic_grads = (
            ppo_critic_loss_grad_function(critic_state.params, traj_batch, targets)
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        total_loss = None
    metrics = (
        total_loss,
        actor_loss,
        critic_loss,
        simple_actor_loss,
        entropy_loss,
        clip_fraction,
        advantages,
        update_targets,
        values,
    )

    return (actor_state, critic_state), metrics


def _update_epoch_pre_partial(  # pylint: disable=R0913, R0914
    update_state: tuple[
        TrainState, TrainState, Transition, jax.Array, jax.Array, random.PRNGKeyArray
    ],
    _: Any,
    minibatch_size: int,
    num_minibatches: int,
    actor_network: Network,
    ent_coef: float,
    clip_coef: float,
    clip_coef_vf: Optional[float],
    critic_network: Network,
    num_steps: int,
    num_envs: int,
    shared_network: bool,
    vf_coef: float,
    advantage_normalization: bool,
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
    rng, permutation_key = jax.random.split(rng)
    batch_size = minibatch_size * num_minibatches
    assert (
        batch_size == num_steps * num_envs
    ), "batch size must be equal to number of steps * number of envs"
    permutation = jax.random.permutation(permutation_key, batch_size)
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

    _update_minibatch = partial(
        _update_minbatch_pre_partial,
        actor_network=actor_network,
        ent_coef=ent_coef,
        clip_coef=clip_coef,
        clip_coef_vf=clip_coef_vf,
        critic_network=critic_network,
        shared_network=shared_network,
        vf_coef=vf_coef,
        advantage_normalization=advantage_normalization,
    )
    (actor_state, critic_state), metrics = jax.lax.scan(
        f=_update_minibatch, init=(actor_state, critic_state), xs=minibatches
    )
    update_state = (
        actor_state,
        critic_state,
        traj_batch,
        advantages,
        targets,
        rng,
    )

    return update_state, metrics


def _update_step_pre_partial(  # pylint: disable=R0913,R0914
    runner_state: tuple[
        TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray
    ],
    _: Any,
    actor_network: Network,
    critic_network: Network,
    env: Environment,
    env_params: EnvParams,
    gamma: float,
    gae_lambda: float,
    minibatch_size: int,
    num_minibatches: int,
    ent_coef: float,
    clip_coef: float,
    clip_coef_vf: Optional[float],
    update_epochs: int,
    num_envs: int,
    num_steps: int,
    log: bool,
    shared_network: bool,
    vf_coef: Optional[float],
    advantage_normalization: bool,
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
        env (Environment): The gymnax environment to train on
        env_params (EnvParams): The gymnax environment parameters.
        gamma (float): The discount factor.
        gae_lambda (float): The gae lambda parameter for advantage computation.
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
        env=env,
        env_params=env_params,
        num_envs=num_envs,
        shared_network=shared_network,
    )
    # COLLECT BATCH TRAJECTORIES
    (actor_state, critic_state, env_state, last_obs, rng), traj_batch = jax.lax.scan(
        f=_env_step, init=runner_state, xs=None, length=num_steps
    )

    # CALCULATE ADVANTAGE

    if shared_network:
        _, last_val = actor_network.apply(actor_state.params, last_obs)
    else:
        last_val = critic_network.apply(critic_state.params, last_obs)

    advantages, targets = _calculate_gae(
        traj_batch, last_val, gamma=gamma, gae_lambda=gae_lambda
    )

    # UPDATE NETWORK
    _update_epoch = partial(
        _update_epoch_pre_partial,
        minibatch_size=minibatch_size,
        num_minibatches=num_minibatches,
        actor_network=actor_network,
        ent_coef=ent_coef,
        clip_coef=clip_coef,
        clip_coef_vf=clip_coef_vf,
        critic_network=critic_network,
        num_steps=num_steps,
        num_envs=num_envs,
        shared_network=shared_network,
        vf_coef=vf_coef,
        advantage_normalization=advantage_normalization,
    )
    (actor_state, critic_state, traj_batch, _, _, rng), metrics = jax.lax.scan(
        f=_update_epoch,
        init=(actor_state, critic_state, traj_batch, advantages, targets, rng),
        xs=None,
        length=update_epochs,
    )

    if log:
        jax.debug.callback(
            wandb_log, traj_batch.info, metrics, num_envs, shared_network
        )

    metric = traj_batch.info

    runner_state = (actor_state, critic_state, env_state, last_obs, rng)
    return runner_state, metric


def make_train(  # pylint: disable=W0102, R0913
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
    log: bool = False,
    env_params: Optional[EnvParams] = None,
    anneal_lr: bool = True,
    max_grad_norm: Optional[float] = 0.5,
    shared_network: bool = False,
    vf_coef: Optional[float] = None,
    advantage_normalization: bool = True,
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

    Returns:
        Callable[ ..., tuple[TrainState, TrainState, EnvState, jax.Array,\
              random.PRNGKeyArray] ]: The train function to be called to actually train.
    """

    num_updates = total_timesteps // num_steps // num_envs
    minibatch_size = num_envs * num_steps // num_minibatches
    if isinstance(env_id, str):
        env_id, env_params = gymnax.make(env_id)
    env = LogWrapper(FlattenObservationWrapper(env_id))

    def train(
        key: random.PRNGKeyArray,
    ) -> tuple[TrainState, TrainState, EnvState, jax.Array, random.PRNGKeyArray]:
        """Train the agent with the given random key and according to config"""
        # INIT NETWORK

        (
            rng,
            actor_key,
            critic_key,
        ) = random.split(key, num=3)
        actor_network, critic_network = init_networks(
            env=env,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            shared_network=shared_network,
            multiple_envs=num_envs > 1,
        )
        if anneal_lr:
            scheduler = get_parameterized_schedule(
                linear_scheduler=annealed_linear_schedule,
                initial_learning_rate=learning_rate,
                num_minibatches=num_minibatches,
                update_epochs=update_epochs,
                num_updates=num_updates,
            )
        else:
            scheduler = learning_rate  # type: ignore[assignment]

        actor_state, critic_state = init_actor_and_critic_state(
            actor_network=actor_network,
            critic_network=critic_network,
            actor_key=actor_key,
            critic_key=critic_key,
            env=env,
            actor_tx=get_adam_tx(learning_rate=scheduler, max_grad_norm=max_grad_norm),
            critic_tx=get_adam_tx(learning_rate=scheduler, max_grad_norm=max_grad_norm),
            env_params=env_params,
            shared_network=shared_network,
        )

        # INIT ENV
        rng, reset_key = jax.random.split(rng)
        reset_rng = jax.random.split(reset_key, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        # TRAIN LOOP

        runner_state = (
            actor_state,
            critic_state,
            env_state,
            obsv,
            rng,
        )
        _update_step = partial(
            _update_step_pre_partial,
            actor_network=actor_network,
            critic_network=critic_network,
            env=env,
            env_params=env_params,
            gamma=gamma,
            gae_lambda=gae_lambda,
            minibatch_size=minibatch_size,
            num_minibatches=num_minibatches,
            ent_coef=ent_coef,
            clip_coef=clip_coef,
            clip_coef_vf=clip_coef_vf,
            update_epochs=update_epochs,
            num_envs=num_envs,
            num_steps=num_steps,
            log=log,
            shared_network=shared_network,
            vf_coef=vf_coef,
            advantage_normalization=advantage_normalization,
        )
        runner_state, _ = jax.lax.scan(
            f=_update_step, init=runner_state, xs=None, length=num_updates
        )

        return runner_state

    return train
