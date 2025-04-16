from functools import partial as partial_functools
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import random
from jax.tree_util import Partial as partial

from jaxppo.environment.environment_interaction import reset_env, step_env
from jaxppo.networks.networks import (
    EnvironmentProperties,
    MaybeRecurrentTrainState,
    MultiCriticState,
    NetworkProperties,
    get_adam_tx,
    get_pi,
    init_actor_and_critic_state,
    init_networks,
    predict_value,
)
from jaxppo.types_rnn import HiddenState
from jaxppo.utils import (
    annealed_linear_schedule,
    check_env_is_gymnax,
    check_update_frequency,
    get_env_observation_shape,
    get_num_actions,
    get_parameterized_schedule,
    prepare_env,
    save_model,
)


class CollectorState(NamedTuple):
    """The variables necessary to interact with the environment and collect the transitions"""

    rng: jax.Array
    env_state: EnvState
    last_obs: jnp.ndarray
    buffer_state: fbx.flat_buffer.TrajectoryBufferState
    actor_state: MaybeRecurrentTrainState
    num_update: int = 0
    timestep: int = 0
    average_reward: float = 0.0
    last_done: Optional[jnp.ndarray] = None


class AgentState(NamedTuple):
    """
    The agent properties to be carried over iterations of environment interaction and updates
    """

    rng: jax.Array
    actor_state: MaybeRecurrentTrainState
    critic_states: Tuple[MaybeRecurrentTrainState] | MaybeRecurrentTrainState
    target_critic_states: Tuple[MaybeRecurrentTrainState] | MaybeRecurrentTrainState
    buffer_state: fbx.flat_buffer.TrajectoryBufferState
    alpha: TrainState  # Temperature parameter


class OptimizerProperties(NamedTuple):
    learning_rate: float | Callable[[int], float]
    max_grad_norm: Optional[float] = None


class BufferProperties(NamedTuple):
    buffer_size: int
    minibatch_size: int = 1


class AlphaProperties(NamedTuple):
    alpha_init: float
    learning_rate: float


def get_buffer(buffer_size: int, batch_size: int, min_length: int = 1):
    return fbx.make_flat_buffer(
        max_length=buffer_size, sample_batch_size=batch_size, min_length=min_length
    )


def create_alpha_train_state(
    alpha_init: float = 0.1,
    learning_rate: float = 3e-4,
) -> TrainState:
    log_alpha = jnp.log(alpha_init)
    params = FrozenDict({"log_alpha": log_alpha})
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=lambda params: jnp.exp(params["log_alpha"]),  # Optional
        params=params,
        tx=tx,
    )


def init_buffer(
    buffer: fbx.trajectory_buffer.TrajectoryBuffer,
    env_args: EnvironmentProperties,
    obsv: jax.Array,
) -> fbx.trajectory_buffer.TrajectoryBufferState:
    action_size = get_num_actions(env_args.env)
    buffer_state = buffer.init(
        {
            "obs": obsv,
            "action": jnp.zeros(
                (
                    (env_args.num_envs, action_size)
                    if env_args.continuous
                    else (env_args.num_envs,)
                ),
                dtype=jnp.float32 if env_args.continuous else jnp.int32,
            ),
            "reward": jnp.zeros((env_args.num_envs,), dtype=jnp.float32),
            "done": jnp.zeros((env_args.num_envs,), dtype=jnp.float32),
            "next_obs": obsv,
        }
    )
    return buffer_state


@partial(
    jax.jit, static_argnames=["env_args", "optimizer_args", "network_args", "buffer"]
)
def init_agent(
    key: jax.Array,
    env_args: EnvironmentProperties,
    optimizer_args: OptimizerProperties,
    network_args: NetworkProperties,
    buffer: fbx.trajectory_buffer.TrajectoryBuffer,
    alpha_args: AlphaProperties,
):
    (
        rng,
        actor_key,
        critic_key,
    ) = random.split(key, num=3)
    actor_network, critic_network = init_networks(env_args, network_args)

    actor_state, critic_state = init_actor_and_critic_state(
        actor_network=actor_network,
        critic_network=tuple(critic_network for _ in range(2)),
        actor_key=actor_key,
        critic_key=critic_key,
        actor_tx=get_adam_tx(**optimizer_args._asdict()),
        critic_tx=get_adam_tx(**optimizer_args._asdict()),
        lstm_hidden_size=network_args.lstm_hidden_size,
        env_args=env_args,
    )

    target_critic_state = critic_state
    observation_shape = get_env_observation_shape(env_args.env, env_args.env_params)
    obsv = jnp.zeros((env_args.num_envs, *observation_shape))
    buffer_state = init_buffer(buffer, env_args, obsv)

    alpha = create_alpha_train_state(*alpha_args)

    return (rng, actor_state, critic_state, target_critic_state, buffer_state, alpha)


@partial(
    jax.jit,
    static_argnames=["recurrent", "bool", "mode", "env_args", "buffer"],
    donate_argnames=["collector_state"],
)
def collect_experience(
    collector_state: CollectorState,
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentProperties,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
):
    pi, new_actor_hidden_state = get_pi(
        collector_state.actor_state.state,
        collector_state.actor_state.state.params,
        (
            collector_state.last_obs[jnp.newaxis, :]
            if recurrent
            else collector_state.last_obs
        ),
        collector_state.actor_state.hidden_state if recurrent else None,
        collector_state.last_done[jnp.newaxis, :] if recurrent else None,  # type: ignore[index]
        recurrent,
    )
    rng, action_key = jax.random.split(collector_state.rng)
    action = pi.sample(seed=action_key)  # cast to float to unify gymnax with brax
    if env_args.continuous:
        action = jnp.float_(action)
    if recurrent:
        action = action.squeeze(0)

    rng, step_key = jax.random.split(rng)

    rng_step = (
        jax.random.split(step_key, env_args.num_envs) if mode == "gymnax" else step_key
    )

    obsv, env_state, reward, done, info = step_env(
        rng_step,
        collector_state.env_state,
        action,
        env_args.env,
        mode,
        env_args.env_params,
    )

    buffer_state = buffer.add(
        collector_state.buffer_state,
        {
            "obs": collector_state.last_obs,
            "action": action,
            "reward": reward,
            "done": collector_state.last_done,
            "next_obs": obsv,
        },
    )

    actor_state = MaybeRecurrentTrainState(
        state=collector_state.actor_state.state, hidden_state=new_actor_hidden_state
    )
    return (
        CollectorState(
            rng=rng,
            env_state=env_state,
            last_obs=obsv,
            buffer_state=buffer_state,
            actor_state=actor_state,
            timestep=collector_state.timestep + 1,
            last_done=done,
        ),
        None,
    )


@partial(jax.jit, static_argnames=["recurrent", "gamma"])
def value_loss_function(
    critic_params: FrozenDict,  # {'critic1': ..., 'critic2': ...}
    critic_states: Tuple[TrainState, TrainState],
    observations: jax.Array,
    next_observations: jax.Array,
    actor_state: MaybeRecurrentTrainState,
    recurrent: bool,
    dones: jax.Array,
    rng: jax.Array,
    rewards: jax.Array,
    target_critic_states: Tuple[MaybeRecurrentTrainState, MaybeRecurrentTrainState],
    gamma: float,
    alpha: jax.Array,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    # Recurrent handling helpers
    def maybe(x):
        return x[jnp.newaxis, :] if recurrent else x

    def unsqueeze_all(xs):
        return tuple(x.squeeze(0) if recurrent else x for x in xs)

    obs, next_obs, done = maybe(observations), maybe(next_observations), maybe(dones)

    # Sample next actions from policy π(a|s_{t+1})
    pi, _ = get_pi(
        actor_state.state,
        actor_state.state.params,
        next_obs,
        actor_state.hidden_state if recurrent else None,
        done if recurrent else None,
        recurrent,
    )
    next_actions = pi.sample(seed=rng)
    log_probs = pi.log_prob(next_actions)

    # Predict Q-values from critics
    q_preds = tuple(
        predict_value(
            cs,
            critic_params[f"critic{i+1}"],
            obs,
            cs.hidden_state if recurrent else None,
            done if recurrent else None,
            recurrent,
            next_actions,
        )[0]
        for i, cs in enumerate(critic_states)
    )

    # Target Q-values using target networks
    q_targets = tuple(
        predict_value(
            ts.state,
            ts.state.params,
            next_obs,
            ts.hidden_state if recurrent else None,
            done if recurrent else None,
            recurrent,
            next_actions,
        )[0]
        for ts in target_critic_states
    )

    # Unpack and unsqueeze if needed
    q1_pred, q2_pred = unsqueeze_all(q_preds)
    q1_target, q2_target = unsqueeze_all(q_targets)
    log_probs = maybe(log_probs)
    next_actions = maybe(next_actions)

    # Bellman target and losses
    min_q_target = jnp.minimum(q1_target, q2_target)
    target_q = rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs)

    loss_q1 = jnp.mean((q1_pred - target_q) ** 2)
    loss_q2 = jnp.mean((q2_pred - target_q) ** 2)
    total_loss = loss_q1 + loss_q2

    aux = dict(
        critic_loss=total_loss,
        q1_loss=loss_q1,
        q2_loss=loss_q2,
        q1_pred=q1_pred,
        q2_pred=q2_pred,
        target_q=target_q,
        log_probs=log_probs,
    )

    return total_loss, aux


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma"],
)
def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: MaybeRecurrentTrainState,
    critic_states: Tuple[MaybeRecurrentTrainState, MaybeRecurrentTrainState],
    observations: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
    alpha: jax.Array,
    rng: jax.random.PRNGKey,
) -> Tuple[jax.Array, Dict[str, Any]]:
    pi, _ = get_pi(
        actor_state.state,
        actor_params,
        observations,
        actor_state.hidden_state if recurrent else None,
        done if recurrent else None,
        recurrent,
    )
    actions = pi.sample(seed=rng)
    log_probs = pi.log_prob(actions)

    q_preds = tuple(
        predict_value(
            cs.state,
            cs.state.params,
            observations,
            cs.hidden_state if recurrent else None,
            done if recurrent else None,
            recurrent,
            actions,
        )[0]
        for cs in critic_states
    )
    q_min = jnp.minimum(*q_preds)
    loss = (alpha * log_probs - q_min).mean()

    return loss, {
        "policy_loss": loss,
        "log_pi": log_probs.mean(),
        "q_min": q_min.mean(),
    }


@partial(jax.jit, static_argnames=["recurrent"])
def update_policy(
    observations: jax.Array,
    done: Optional[jax.Array],
    agent_state: AgentState,
    recurrent: bool,
) -> Tuple[AgentState, Dict[str, Any]]:
    rng, policy_key = jax.random.split(agent_state.rng)
    value_and_grad_fn = jax.value_and_grad(policy_loss_function, has_aux=True)

    (loss, aux), grads = value_and_grad_fn(
        agent_state.actor_state.state.params,
        agent_state.actor_state,
        agent_state.critic_states,
        observations,
        done,
        recurrent,
        agent_state.alpha,
        policy_key,
    )

    updated_actor_state = agent_state.actor_state.state.apply_gradients(grads=grads)
    new_actor_state = agent_state.actor_state.replace(state=updated_actor_state)

    return (
        AgentState(
            rng=rng,
            actor_state=new_actor_state,
            critic_states=agent_state.critic_states,
            target_critic_states=agent_state.target_critic_states,
            buffer_state=agent_state.buffer_state,
            alpha=agent_state.alpha,
        ),
        aux,
    )


@partial(
    jax.jit,
    static_argnames=["recurrent", "gamma"],
)
def update_value_functions(
    observations: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    agent_state: AgentState,
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
) -> Tuple[AgentState, Dict[str, Any]]:
    value_loss_key, rng = jax.random.split(agent_state.rng)
    value_and_grad_fn = jax.value_and_grad(value_loss_function, has_aux=True)

    critic_params = FrozenDict(
        {
            f"critic{i+1}": cs.state.params
            for i, cs in enumerate(agent_state.critic_states)
        }
    )

    critic_states = tuple(cs.state for cs in agent_state.critic_states)

    (loss, aux), grads = value_and_grad_fn(
        critic_params,
        critic_states,
        observations,
        next_observations,
        agent_state.actor_state,
        recurrent,
        dones,
        value_loss_key,
        rewards,
        agent_state.target_critic_states,
        gamma,
        alpha,
    )

    updated_critic_states = tuple(
        cs.replace(state=cs.state.apply_gradients(grads=grads[f"critic{i+1}"]))
        for i, cs in enumerate(agent_state.critic_states)
    )

    return (
        AgentState(
            rng=rng,
            actor_state=agent_state.actor_state,
            critic_states=updated_critic_states,
            target_critic_states=agent_state.target_critic_states,
            buffer_state=agent_state.buffer_state,
            alpha=agent_state.alpha,
        ),
        aux,
    )


@partial(
    jax.jit,
    static_argnames=["target_entropy"],
)
def alpha_loss_function(
    log_alpha_params: FrozenDict,
    log_probs: jax.Array,
    target_entropy: float,
) -> Tuple[jax.Array, Dict[str, Any]]:
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = (alpha * (-log_probs - target_entropy)).mean()

    return loss, {
        "alpha_loss": loss,
        "alpha": alpha,
        "log_alpha": log_alpha,
    }


@partial(
    jax.jit,
    static_argnames=["target_entropy", "recurrent"],
)
def update_temperature(
    agent_state: AgentState,
    observations: jax.Array,
    target_entropy: float,
    recurrent: bool,
    done: bool,
) -> Tuple[AgentState, Dict[str, Any]]:
    loss_fn = jax.value_and_grad(alpha_loss_function, has_aux=True)
    pi, _ = get_pi(
        agent_state.actor_state.state,
        agent_state.actor_state.state.params,
        observations,
        agent_state.actor_state.hidden_state if recurrent else None,
        done if recurrent else None,
        recurrent,
    )
    rng, sample_key = jax.random.split(agent_state.rng)
    actions = pi.sample(seed=sample_key)
    log_probs = pi.log_prob(actions)
    (loss, aux), grads = loss_fn(
        agent_state.alpha.params,
        log_probs,
        target_entropy,
    )

    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)

    return (
        AgentState(
            rng=rng,
            actor_state=agent_state.actor_state,
            critic_states=agent_state.critic_states,
            target_critic_states=agent_state.target_critic_states,
            buffer_state=agent_state.buffer_state,
            alpha=new_alpha_state,
        ),
        aux,
    )


@partial(
    jax.jit,
    static_argnames=["buffer"],
)
def get_batch_from_buffer(buffer, buffer_state, key):
    batch = buffer.sample(buffer_state, key).experience
    return batch.first.obs, batch.first.done, batch.first.next_obs, batch.first.rewards


@partial(
    jax.jit,
    static_argnames=["env_args", "mode", "recurrent", "buffer"],
    donate_argnames=["agent_state", "collector_state"],
)
def update_agent(
    agent_state: AgentState,
    collector_state: CollectorState,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    recurrent: bool,
    gamma: float,
    action_dim: int,
) -> AgentState:
    # Sample buffer

    sample_key, rng = jax.random.split(agent_state.rng)
    observations, dones, next_observations, rewards = get_batch_from_buffer(
        buffer, collector_state.buffer_state, sample_key
    )

    # Update Q functions
    log_alpha = agent_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    agent_state, aux_value = update_value_functions(
        observations=observations,
        next_observations=next_observations,
        dones=dones,
        agent_state=agent_state,
        recurrent=recurrent,
        rewards=rewards,
        gamma=gamma,
        alpha=alpha,
    )

    # Update policy
    agent_state, aux_policy = update_policy(
        observations=observations,
        next_observations=next_observations,
        dones=dones,
        agent_state=agent_state,
        recurrent=recurrent,
        rewards=rewards,
        gamma=gamma,
    )

    # Adjust temperature
    target_entropy = -float(action_dim)
    agent_state, aux_temperature = update_temperature(
        agent_state,
        observations=observations,
        target_entropy=target_entropy,
        recurrent=recurrent,
        done=dones,
    )

    # Update target networks
    agent_state = update_target_networks(agent_state)

    return agent_state


@partial(
    jax.jit,
    static_argnames=["env_args", "mode", "recurrent", "buffer"],
    donate_argnames=["agent_state"],
)
def training_iteration(
    agent_state: AgentState,
    _: Any,
    env_args: EnvironmentProperties,
    mode: str,
    recurrent: bool,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
):
    """
    Run one iteration of the algorithm : Collect experience from the environment and use it to update the agent.
    """
    last_done = jnp.zeros(env_args.num_envs)
    reset_keys = (
        jax.random.split(agent_state.rng, env_args.num_envs)
        if mode == "gymnax"
        else agent_state.rng
    )
    last_obs, env_state = reset_env(reset_keys, env_args.env, mode, env_args.env_params)
    collector_state = CollectorState(
        rng=agent_state.rng,
        env_state=env_state,
        last_obs=last_obs,
        buffer_state=agent_state.buffer_state,
        actor_state=agent_state.actor_state,
        timestep=0,
        last_done=last_done,
    )
    scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
    )
    collector_state, _ = jax.lax.scan(scan_fn, collector_state, xs=None, length=1)

    agent_state = update_agent(agent_state, collector_state)
    return agent_state, None


def make_train(
    env_args: EnvironmentProperties,
    optimizer_args: OptimizerProperties,
    network_args: NetworkProperties,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
):
    def train(key, total_timesteps):
        (
            rng,
            actor_state,
            critic_state,
            target_critic_state,
            buffer_state,
        ) = init_agent(
            key=key,
            env_args=env_args,
            optimizer_args=optimizer_args,
            network_args=network_args,
            buffer=buffer,
        )

        agent_state = AgentState(
            rng=rng,
            actor_state=actor_state,
            critic_states=critic_state,
            target_critic_states=target_critic_state,
            buffer_state=buffer_state,
        )

        num_updates = total_timesteps // env_args.num_envs

        agent_state, _ = jax.lax.scan(
            f=training_iteration, init=agent_state, xs=None, length=num_updates
        )

    return train


# def train_sac():

#     def environment_step():
#         sample_action_from_policy()
#         sample_transition_from_environment()
#         add_transition_to_buffer()

#     def update_step():
#         batch = sample_buffer()
#         update_q_functions()
#         update_policy()
#         adjust_temperature()
#         update_target_network_weights()

#     def iteration():
#         environment_step()
#         update_step()

#     for iter in num_iterations:
#         iteration()


if __name__ == "__main__":
    train()
