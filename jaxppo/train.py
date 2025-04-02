"""Pure jax PPO training"""

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import random
from jax.tree_util import Partial as partial

from jaxppo.evaluate import evaluate
from jaxppo.networks.networks import (
    get_adam_tx,
    get_pi,
    init_actor_and_critic_state,
    init_networks,
    predict_value,
)
from jaxppo.types_rnn import BatchInfo, HiddenState
from jaxppo.utils import (
    annealed_linear_schedule,
    check_env_is_gymnax,
    check_update_frequency,
    get_num_actions,
    get_parameterized_schedule,
    prepare_env,
    save_model,
)
from jaxppo.video import save_video_to_wandb
from jaxppo.wandb_logging import log_variables


class Transition(NamedTuple):
    """Store transitions inside a buffer"""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class RunnerState(NamedTuple):
    """The current agent state used to step the env"""

    actor_state: TrainState
    env_state: EnvState
    last_obs: jnp.ndarray
    rng: jax.Array
    critic_state: TrainState
    last_done: Optional[jnp.ndarray] = None
    actor_hidden_state: Optional[jnp.ndarray] = None
    critic_hidden_state: Optional[jnp.ndarray] = None
    num_update: int = 0
    timestep: int = 0
    average_reward: float = 0.0


class UpdateState(NamedTuple):
    """The current state of the updated parameters and the required variables\
          for the update"""

    actor_state: TrainState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jax.Array
    critic_state: TrainState
    actor_hidden_state: Optional[HiddenState] = None
    critic_hidden_state: Optional[HiddenState] = None


@partial(jax.jit, static_argnames=["average_reward_mode"])
def _calculate_gae(
    traj_batch: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
    last_done: Optional[jax.Array] = None,
    average_reward: Optional[jax.Array] = None,
    average_reward_mode: bool = False,
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
        gae_next_value_and_next_done: tuple[jax.Array, jax.Array, Optional[jax.Array]],
        transition: Transition,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
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
        gae, next_value, next_done = gae_next_value_and_next_done
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        if average_reward_mode:
            assert average_reward is not None
            target = reward - average_reward + next_value
            delta = target - value
            gae = delta  # + gae * gae_lambda
        else:
            next_state_is_non_terminal = (
                1.0 - done if next_done is None else 1.0 - next_done
            )

            delta = reward + gamma * next_value * next_state_is_non_terminal - value
            gae = delta + gamma * gae_lambda * next_state_is_non_terminal * gae

        # tuple is carry-over state for scan, gae after the comma is the actual return at the end of the scan
        return (gae, value, done if next_done is not None else None), gae

    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=(jnp.zeros_like(last_val), last_val, last_done),
        xs=traj_batch,
        reverse=True,
        unroll=16,
    )
    returns = advantages + traj_batch.value
    return advantages, returns


def init_agent(  # pylint: disable=W0102, R0913
    key: jax.Array,
    env: Environment,
    actor_architecture: list[str],
    critic_architecture: list[str],
    num_envs: int,
    anneal_lr: float,
    learning_rate: float,
    num_minibatches: int,
    update_epochs: int,
    num_updates: int,
    max_grad_norm: Optional[float],
    env_params: EnvParams,
    lstm_hidden_size: Optional[int] = None,
    continuous: bool = False,
):
    """Initialize the networks, actor/critic states, and rng"""
    (
        rng,
        actor_key,
        critic_key,
    ) = random.split(key, num=3)
    actor_network, critic_network = init_networks(
        env=env,
        params=env_params,
        actor_architecture=actor_architecture,
        critic_architecture=critic_architecture,
        multiple_envs=num_envs > 1,
        lstm_hidden_size=lstm_hidden_size,
        continuous=continuous,
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

    (actor_state, critic_state), (actor_hidden_state, critic_hidden_state) = (
        init_actor_and_critic_state(
            actor_network=actor_network,
            critic_network=critic_network,
            actor_key=actor_key,
            critic_key=critic_key,
            env=env,
            actor_tx=get_adam_tx(learning_rate=scheduler, max_grad_norm=max_grad_norm),
            critic_tx=get_adam_tx(learning_rate=scheduler, max_grad_norm=max_grad_norm),
            env_params=env_params,
            lstm_hidden_size=lstm_hidden_size,
            num_envs=num_envs,
        )
    )
    return (
        actor_state,
        critic_state,
        actor_hidden_state,
        critic_hidden_state,
        rng,
    )


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
    advantage_normalization: bool = True,
    save: bool = False,
    save_folder: str = "./models",
    num_eval_envs: int = 4,
    log_video: bool = False,
    video_log_frequency: Optional[int] = None,
    save_frequency: Optional[int] = None,
    lstm_hidden_size: Optional[int] = None,
    continuous: bool = False,
    average_reward_mode: bool = False,
    window_size: int = 32,
    episode_length: int = Optional[None],
    render_env_id: Optional[str] = None,
) -> Callable[..., RunnerState]:
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
              jax.Array] ]: The train function to be called to actually train.
    """
    num_updates = total_timesteps // num_steps // num_envs
    minibatch_size = num_envs * num_steps // num_minibatches
    env, env_params, env_id = prepare_env(
        env_id, continuous, gamma, episode_length, env_params=env_params
    )
    num_actions = get_num_actions(env, env_params)
    mode = "gymnax" if check_env_is_gymnax(env) else "brax"
    if log_video:
        if not isinstance(env_id, str):
            if "render" not in dir(env):
                raise AttributeError(
                    f"Environment {env} has no render method yet video saving is"
                    " enabled."
                )

    @jax.jit
    def train(
        key: jax.Array,
    ) -> RunnerState:
        """Train the agent with the given random key and according to config"""
        # INIT NETWORK

        (
            actor_state,
            critic_state,
            actor_hidden_state,
            critic_hidden_state,
            rng,
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
            lstm_hidden_size=lstm_hidden_size,
            continuous=continuous,
        )
        # INIT ENV
        rng, reset_key = jax.random.split(rng)
        reset_rng = jax.random.split(reset_key, num_envs)

        def reset_env(
            env: Environment,
            rng: jax.Array,
            env_params: Optional[EnvParams] = None,
        ) -> tuple[jax.Array, EnvState]:
            if mode == "gymnax":
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
                    rng, env_params
                )
            else:  # brax
                env_state = jax.vmap(jax.jit(env.reset), in_axes=(0))(rng)
                obsv = env_state.obs
            return obsv, env_state

        def step_env(env, state, action, rng, env_params):
            if mode == "gymnax":
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng, state, action, env_params)
                done = jnp.float_(done)  # cast to float to unify with brax
            else:
                env_state = jax.vmap(env.step, in_axes=(0, 0))(state, action)
                obsv, reward, done, info = (
                    env_state.obs,
                    env_state.reward,
                    env_state.done,
                    env_state.info,
                )

            return obsv, env_state, reward, done, info

        obsv, env_state = reset_env(env, reset_rng, env_params)
        # TRAIN LOOP
        last_done = jnp.zeros(num_envs, dtype=float)
        timestep = 0
        runner_state = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
            timestep=timestep,
            actor_hidden_state=actor_hidden_state,
            critic_hidden_state=critic_hidden_state,
            last_done=last_done,
        )
        recurrent = actor_hidden_state is not None and critic_hidden_state is not None

        def _update_step_pre_partial(  # pylint: disable=R0913,R0914
            runner_state: RunnerState,
            _: Any,
        ) -> tuple[RunnerState, dict]:
            """
            Update the agent (actor and critic network states) on the provided environment(s)\
                for the given number of epochs, num of minibatches and minibatch size.

            Args:
                runner_state (tuple[ TrainState, TrainState, EnvState, jax.Array,\
                    jax.Array ]): The runner state (containing, Actor, Critic\
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
                save (bool): Wether or not to save model

            Returns:
            tuple[
                tuple[TrainState, TrainState, EnvState, jax.Array, \
                jax.Array], dict]:\
                    The new runner state (containing, Actor, Critic\
                        and Env states, your last observation and the new random key).
                    and a metric dict.
            """

            def _env_step_pre_partial(
                runner_state: RunnerState,
                _: Any,
            ) -> tuple[RunnerState, Transition]:
                """
                Step the environment (over num_envs envs) and collect the transitions the buffer\
                    according to the current actor, critic and env states. Then updates env state

                Args:
                    runner_state (tuple[ TrainState, TrainState, EnvState, jax.Array, \
                        jax.Array ]): Current agent state \
                            (actor state, critic state, env state)
                    _ (Any): Blank parameter for vectorization
                    actor_network (Network): The actor network to use.
                    critic_network (Network): The critic network to use.
                    action_key (jax.Array): The random key to be used in action sampling.
                    step_key (jax.Array): The random key to be used in env stepping.
                    env (Environment): The gymnax environment to step over.
                    env_params (EnvParams): The gymnax environment parameters.
                    num_envs (int): The number of environments to use.

                Returns:
                    tuple[ tuple[TrainState, TrainState, EnvState, jax.Array, jax.Array],\
                        Transition ]: The new runner state \
                            (actor state, critic state and env state)\
                            and the Transition buffer
                """
                # SELECT ACTION
                pi, new_actor_hidden_state = get_pi(
                    runner_state.actor_state,
                    runner_state.actor_state.params,
                    (
                        runner_state.last_obs[jnp.newaxis, :]
                        if recurrent
                        else runner_state.last_obs
                    ),
                    runner_state.critic_hidden_state if recurrent else None,
                    runner_state.last_done[jnp.newaxis, :] if recurrent else None,  # type: ignore[index]
                    recurrent,
                )
                value, new_critic_hidden_state = predict_value(
                    runner_state.critic_state,
                    runner_state.critic_state.params,
                    (
                        runner_state.last_obs[jnp.newaxis, :]
                        if recurrent
                        else runner_state.last_obs
                    ),
                    runner_state.critic_hidden_state if recurrent else None,
                    runner_state.last_done[jnp.newaxis, :] if recurrent else None,  # type: ignore[index]
                    recurrent,
                )
                rng, action_key = jax.random.split(runner_state.rng)
                # pi = runner_state.actor_state.apply_fn(
                #     runner_state.actor_state.params, runner_state.last_obs
                # )
                # if not recurrent:
                #     # action = pi.sample(seed=action_key).reshape(
                #     #     -1,
                #     # )
                #     action = pi.sample(seed=action_key)  # cast to float for consistency
                # else:
                action = jnp.float_(
                    pi.sample(seed=action_key)
                )  # cast to float to unify gymnax with brax
                log_prob = pi.log_prob(action)

                if recurrent:
                    value, action, log_prob = (
                        value.squeeze(0),
                        action.squeeze(0),
                        log_prob.squeeze(0),
                    )

                # STEP ENV
                rng, step_key = jax.random.split(rng)
                rng_step = jax.random.split(step_key, num_envs)

                obsv, env_state, reward, done, info = step_env(
                    env, runner_state.env_state, action, rng_step, env_params
                )
                transition = Transition(
                    runner_state.last_done,
                    action,
                    value,
                    reward,
                    log_prob,
                    runner_state.last_obs,
                    info,
                )
                runner_state = RunnerState(
                    actor_state=runner_state.actor_state,
                    critic_state=runner_state.critic_state,
                    env_state=env_state,
                    last_obs=obsv,
                    rng=rng,
                    actor_hidden_state=new_actor_hidden_state,
                    critic_hidden_state=new_critic_hidden_state,
                    last_done=done,
                )
                return runner_state, transition

            initial_actor_hidden_state = runner_state.actor_hidden_state
            initial_critic_hidden_state = runner_state.critic_hidden_state
            if recurrent:
                if (
                    initial_actor_hidden_state is None
                    or initial_critic_hidden_state is None
                ):
                    raise ValueError(
                        "Hidden states are None in recurrent mode"
                        f" {initial_actor_hidden_state=},"
                        f" {initial_critic_hidden_state=}"
                    )

            timestep = runner_state.timestep
            num_update = runner_state.num_update
            # COLLECT BATCH TRAJECTORIES
            runner_state, traj_batch = jax.lax.scan(
                f=_env_step_pre_partial, init=runner_state, xs=None, length=num_steps
            )
            avg = None

            if average_reward_mode:
                avg = jnp.average(
                    traj_batch.reward[: -min(window_size, num_steps - 1)],
                    axis=0,
                )
            # CALCULATE ADVANTAGE
            last_val, _ = predict_value(
                runner_state.critic_state,
                runner_state.critic_state.params,
                (
                    runner_state.last_obs[jnp.newaxis, :]
                    if recurrent
                    else runner_state.last_obs
                ),
                runner_state.critic_hidden_state if recurrent else None,
                runner_state.last_done[jnp.newaxis, :] if recurrent else None,  # type: ignore[index]
                recurrent,
            )
            if recurrent:
                last_val = last_val.squeeze(0)
            advantages, targets = _calculate_gae(
                traj_batch,
                last_val,
                last_done=runner_state.last_done,
                gamma=gamma,
                gae_lambda=gae_lambda,
                average_reward_mode=average_reward_mode,
                average_reward=avg,
            )

            # UPDATE NETWORK
            def _update_epoch_pre_partial(  # pylint: disable=R0913, R0914
                update_state: UpdateState,
                _: Any,
            ) -> tuple[UpdateState, tuple[jax.Array, jax.Array, jax.Array, jax.Array],]:
                """
                Update the actor and critic states over an epoch of num_minibatches minibatches.\
                    Collect losses for logging.

                Args:
                    update_state (tuple[ TrainState, TrainState, Transition, jax.Array, jax.Array,\
                    jax.Array ]): \
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
                    jax.Array ], tuple[jax.Array, jax.Array, jax.Array, jax.Array], ]:\
                        The new update state (The states of the actor and critic networks, the\
                        transition buffer for this epoch, the advantages and target values for\
                        this epoch and the permutation random key) and losses for logging.
                """
                rng, permutation_key = jax.random.split(update_state.rng)
                batch_size = minibatch_size * num_minibatches
                assert (
                    batch_size == num_steps * num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(
                    permutation_key, num_envs if recurrent else batch_size
                )
                # Not possible to permutate differently that the env order, otherwise it messes up hidden states
                if recurrent:
                    batch = (
                        update_state.traj_batch,
                        update_state.advantages,
                        update_state.targets,
                        update_state.actor_hidden_state,
                        update_state.critic_hidden_state,
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], num_minibatches, -1] + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )
                else:
                    batch = (  # type: ignore[assignment]
                        update_state.traj_batch,
                        update_state.advantages,
                        update_state.targets,
                    )
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x, [num_minibatches, -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )

                def _update_minbatch_pre_partial(  # pylint: disable=R0914
                    train_state: tuple[TrainState, TrainState],
                    batch_info: tuple[Transition, jax.Array, jax.Array],
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
                    if recurrent:
                        _, advantages, _, _, _ = batch_info  # type: ignore[misc]
                    else:
                        _, advantages, _ = batch_info  # type: ignore[misc]

                    actor_state, critic_state = train_state

                    def _actor_loss_fn_pre_partial(
                        actor_params: FrozenDict,
                        actor_state: TrainState,
                        batch_info: BatchInfo,
                        gae: jax.Array,
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
                        if recurrent:
                            traj_batch, _, _, actor_hidden_state, _ = batch_info  # type: ignore[misc]
                        else:
                            traj_batch, _, _ = batch_info  # type: ignore[misc]
                        # RERUN NETWORK
                        pi, _ = get_pi(
                            actor_state,
                            actor_params,
                            traj_batch.obs,
                            actor_hidden_state[0] if recurrent else None,
                            traj_batch.done if recurrent else None,
                            recurrent,
                        )

                        # pi = actor_state.apply_fn(actor_params, traj_batch.obs)
                        log_prob = pi.log_prob(
                            traj_batch.action.reshape(-1, num_actions)
                            if continuous
                            else traj_batch.action
                        )
                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        if advantage_normalization:
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        if (num_envs > 1) and continuous:
                            gae = gae.reshape(-1, 1)
                        assert (
                            ratio.shape == (minibatch_size, num_actions)
                            if continuous
                            else (minibatch_size,)
                        )
                        assert ratio.shape[0] == gae.shape[0], (
                            f"Mismatch between ratio shape ({ratio.shape}) and gae"
                            f" shape ({gae.shape})"
                        )
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - clip_coef,
                                1.0 + clip_coef,
                            )
                            * gae
                        )

                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        # CALCULATE AUXILIARIES
                        clip_fraction = (jnp.abs(ratio - 1) > clip_coef).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor - ent_coef * entropy
                        auxiliaries = (
                            loss_actor,
                            entropy,
                            jax.lax.stop_gradient(clip_fraction),
                        )
                        return total_loss, auxiliaries

                    ppo_actor_loss_grad_function = jax.value_and_grad(
                        _actor_loss_fn_pre_partial, has_aux=True
                    )

                    def _critic_loss_fn_pre_partial(
                        critic_params: FrozenDict,
                        critic_state: TrainState,
                        batch_info: BatchInfo,
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
                        if recurrent:
                            traj_batch, _, targets, _, critic_hidden_state = batch_info  # type: ignore[misc]
                        else:
                            traj_batch, _, targets = batch_info  # type: ignore[misc]
                        # RERUN NETWORK
                        value, _ = predict_value(
                            critic_state,
                            critic_params,
                            traj_batch.obs,
                            critic_hidden_state[0] if recurrent else None,
                            traj_batch.done if recurrent else None,
                            recurrent,
                        )
                        value_losses = (value - targets) ** 2

                        # CALCULATE VALUE LOSS
                        if clip_coef_vf is not None:
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-clip_coef_vf, clip_coef_vf)
                            value_losses_clipped = jnp.square(
                                value_pred_clipped - targets
                            )
                            return 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean(), (
                                targets.mean(),
                                value_pred_clipped.mean(),
                            )
                        value_loss = 0.5 * value_losses.mean()
                        auxiliaries = (
                            jax.lax.stop_gradient(targets).mean(),
                            jax.lax.stop_gradient(value).mean(),
                        )
                        return value_loss, auxiliaries

                    ppo_critic_loss_grad_function = jax.value_and_grad(
                        _critic_loss_fn_pre_partial, has_aux=True
                    )
                    (
                        actor_loss,
                        (simple_actor_loss, entropy_loss, clip_fraction),
                    ), actor_grads = ppo_actor_loss_grad_function(
                        actor_state.params, actor_state, batch_info, advantages
                    )

                    (critic_loss, (update_targets, values)), critic_grads = (
                        ppo_critic_loss_grad_function(
                            critic_state.params, critic_state, batch_info
                        )
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

                (actor_state, critic_state), metrics = jax.lax.scan(
                    f=_update_minbatch_pre_partial,
                    init=(update_state.actor_state, update_state.critic_state),
                    xs=minibatches,
                )

                update_state = UpdateState(
                    actor_state=actor_state,
                    critic_state=critic_state,
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    rng=rng,
                    actor_hidden_state=(
                        update_state.actor_hidden_state if recurrent else None
                    ),
                    critic_hidden_state=(
                        update_state.critic_hidden_state if recurrent else None
                    ),
                )

                return update_state, metrics

            if recurrent:
                initial_actor_hidden_state, initial_critic_hidden_state = (
                    initial_actor_hidden_state[None, :],  # type: ignore[index]
                    initial_critic_hidden_state[None, :],  # type: ignore[index]
                )
            update_state = UpdateState(
                actor_state=runner_state.actor_state,
                critic_state=runner_state.critic_state,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=runner_state.rng,
                actor_hidden_state=initial_actor_hidden_state,
                critic_hidden_state=initial_critic_hidden_state,
            )
            update_state, _ = jax.lax.scan(
                f=_update_epoch_pre_partial,
                init=update_state,
                xs=None,
                length=update_epochs,
            )

            # Evaluation over num eval envs

            timestep += num_envs * num_steps
            if log:
                eval_rewards, eval_entropy = evaluate(
                    env,
                    update_state.actor_state,
                    num_eval_envs,
                    update_state.rng,
                    env_params,
                    recurrent=recurrent,
                    lstm_hidden_size=lstm_hidden_size,
                )

                rewards_to_log = {
                    f"episodic reward/reward-{ii}": reward
                    for ii, reward in enumerate(eval_rewards)
                }
                rewards_to_log["timestep"] = timestep
                jax.debug.callback(log_variables, rewards_to_log)
                entropy_to_log = {
                    f"episodic entropy/entropy-{ii}": entropy
                    for ii, entropy in enumerate(eval_entropy)
                }
                entropy_to_log["timestep"] = timestep
                jax.debug.callback(log_variables, entropy_to_log)

            metric = traj_batch.info
            runner_state = RunnerState(
                actor_state=update_state.actor_state,
                critic_state=update_state.critic_state,
                env_state=runner_state.env_state,
                last_obs=runner_state.last_obs,
                rng=update_state.rng,
                timestep=timestep,
                num_update=num_update + 1,
                actor_hidden_state=runner_state.actor_hidden_state,
                critic_hidden_state=runner_state.critic_hidden_state,
                last_done=runner_state.last_done,
            )
            if log_video:
                # Save model and video if needed
                _save_video_to_wandb = partial(
                    save_video_to_wandb, render_env_id, env, recurrent
                )

                def save_video_callback(update_state, params):
                    jax.debug.callback(
                        _save_video_to_wandb, update_state, update_state.rng, params
                    )

                record_video_flag = check_update_frequency(
                    runner_state.num_update, num_updates, video_log_frequency
                )
                jax.lax.cond(
                    record_video_flag,
                    save_video_callback,
                    lambda update_state, params: None,
                    update_state,
                    env_params,
                )

            if save:
                _save_model = partial(save_model, save_folder=save_folder)

                def save_model_cond(actor, critic, index, log):
                    jax.debug.callback(_save_model, actor, critic, index, log)

                jax.lax.cond(
                    check_update_frequency(
                        runner_state.num_update, num_updates, save_frequency
                    ),
                    save_model_cond,
                    lambda actor, critic, index, log: None,
                    update_state.actor_state,
                    update_state.critic_state,
                    runner_state.num_update,
                    log,
                )

            return runner_state, metric

        runner_state, _ = jax.lax.scan(
            f=_update_step_pre_partial, init=runner_state, xs=None, length=num_updates
        )

        return runner_state

    return train
