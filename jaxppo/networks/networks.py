from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, TypeAlias, Union

import jax
import numpy as np
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams
from optax import GradientTransformation, GradientTransformationExtraArgs

from jaxppo.networks.networks_classic import Network as NetworkClassic
from jaxppo.networks.networks_classic import get_pi as get_pi_classic
from jaxppo.networks.networks_classic import (
    init_actor_and_critic_state as init_actor_and_critic_state_classic,
)
from jaxppo.networks.networks_classic import init_networks as init_networks_classic
from jaxppo.networks.networks_classic import predict_probs as predict_probs_classic
from jaxppo.networks.networks_classic import predict_value as predict_value_classic
from jaxppo.networks.networks_RNN import NetworkRNN
from jaxppo.networks.networks_RNN import get_pi as get_pi_rnn
from jaxppo.networks.networks_RNN import (
    init_actor_and_critic_state as init_actor_and_critic_state_rnn,
)
from jaxppo.networks.networks_RNN import init_networks as init_networks_rnn
from jaxppo.networks.networks_RNN import predict_probs as predict_probs_rnn
from jaxppo.networks.networks_RNN import predict_value_and_hidden
from jaxppo.networks.utils import ActivationFunction
from jaxppo.types_rnn import HiddenState

Network: TypeAlias = Union[NetworkRNN, NetworkClassic]


class NetworkProperties(NamedTuple):
    actor_architecture: Sequence[str]
    critic_architecture: Sequence[str]
    lstm_hidden_size: Optional[int] = None
    action_value: bool = False


class EnvironmentProperties(NamedTuple):
    env: Environment
    env_params: EnvParams
    num_envs: int
    continuous: bool


@struct.dataclass
class MaybeRecurrentTrainState:
    state: TrainState
    hidden_state: Optional[HiddenState] = None


@struct.dataclass
class MultiCriticState:
    critic_states: Tuple[MaybeRecurrentTrainState, ...]


def check_done_and_hidden(done: Optional[np.ndarray], hidden: Optional[HiddenState]):
    """Make sure hidden and done are provided in recurrent mode"""
    if done is None or hidden is None:
        raise ValueError(
            "If using recurrent, done and hidden should be provided. Got"
            f" {done=} and {hidden=}"
        )


@partial(jax.jit, static_argnames="recurrent")
def predict_probs(
    actor_state: TrainState,
    actor_params: FrozenDict,
    obs: np.ndarray,
    hidden: Optional[HiddenState] = None,
    done: Optional[np.ndarray] = None,
    recurrent: bool = False,
) -> jax.Array:
    """Return the predicted action logits of the given obs with the current actor state"""
    if recurrent:
        jax.debug.callback(check_done_and_hidden, done, hidden)
        return predict_probs_rnn(actor_state, actor_params, hidden, obs, done)
    return predict_probs_classic(actor_state, actor_params, obs)


@partial(jax.jit, static_argnames=["recurrent", "is_sequence"])
def predict_value(
    critic_state: TrainState,
    critic_params: FrozenDict,
    obs: np.ndarray,
    hidden: Optional[HiddenState] = None,
    done: Optional[np.ndarray] = None,
    recurrent: bool = False,
    action: Optional[np.ndarray] = None,
) -> tuple[jax.Array, Optional[HiddenState]]:
    """Return the predicted value of the given obs with the current critic state"""
    if recurrent:
        jax.debug.callback(check_done_and_hidden, done, hidden)
        return predict_value_and_hidden(
            critic_state,
            critic_params,
            hidden,
            obs,
            done,
            action,
        )
    return (
        predict_value_classic(critic_state, critic_params, obs, action),
        None,
    )


def get_adam_tx(
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    max_grad_norm: Optional[float] = 0.5,
    eps: float = 1e-5,
    clipped=True,
) -> GradientTransformationExtraArgs:
    """Return a clipped adam optimiser with the given parameters"""
    if clipped and max_grad_norm is None:
        raise ValueError("Gradient clipping requested but no norm provided.")
    if clipped and max_grad_norm is not None:
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, eps=eps),
        )
    return optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, eps=eps)


def init_actor_and_critic_state(
    env_args: EnvironmentProperties,
    actor_network: Union[
        Union[NetworkClassic, NetworkRNN], Sequence[Union[NetworkClassic, NetworkRNN]]
    ],
    actor_key: jax.Array,
    actor_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_network: Union[
        Union[NetworkClassic, NetworkRNN], Sequence[Union[NetworkClassic, NetworkRNN]]
    ],
    critic_key: Optional[jax.Array] = None,
    lstm_hidden_size: Optional[int] = None,
    action_value: bool = False,
) -> tuple[tuple[MaybeRecurrentTrainState, MaybeRecurrentTrainState]]:
    """Initialize the actor and critic train state (and hidden states if needed) given\
          their networks."""
    if lstm_hidden_size is not None:
        if env_args.num_envs is None or lstm_hidden_size is None:
            raise ValueError(
                "Num envs and lstm hidden size should be defined when using rnn, got"
                f" {env_args.num_envs=} and {lstm_hidden_size=}"
            )
        if not (
            isinstance(actor_network, NetworkRNN)
            and isinstance(critic_network, (NetworkRNN, Tuple))
        ):
            raise ValueError(
                "Networks should be recurrent when using recurrent mode, got"
                f" {actor_network=} and {critic_network=}"
            )
        (actor_state, critic_state), (actor_hidden_state, critic_hidden_state) = (
            init_actor_and_critic_state_rnn(
                env_args.env,
                env_args.env_params,
                actor_key,
                env_args.num_envs,
                actor_network,
                actor_tx,
                critic_tx,
                critic_network,
                lstm_hidden_size,
                action_value,
            )
        )
        return MaybeRecurrentTrainState(actor_state, actor_hidden_state), (
            tuple(
                MaybeRecurrentTrainState(cs, h)
                for cs, h in zip(critic_state, critic_hidden_state)
            )
            if isinstance(critic_state, Tuple)
            else MaybeRecurrentTrainState(critic_state, critic_hidden_state)
        )
    actor_state, critic_state = init_actor_and_critic_state_classic(
        env_args.env,
        env_args.env_params,
        actor_network,
        actor_key,
        actor_tx,
        critic_tx,
        critic_network,
        critic_key,
        action_value=action_value,
    )
    return MaybeRecurrentTrainState(actor_state), (
        tuple(MaybeRecurrentTrainState(cs) for cs in critic_state)
        if isinstance(critic_state, Tuple)
        else MaybeRecurrentTrainState(critic_state)
    )


def init_networks(
    env_args: EnvironmentProperties,
    network_args: NetworkProperties,
) -> Union[tuple[NetworkClassic, NetworkClassic], tuple[NetworkRNN, NetworkRNN]]:
    multiple_envs = env_args.num_envs > 1
    if network_args.lstm_hidden_size is not None:
        return init_networks_rnn(
            env=env_args.env,
            params=env_args.env_params,
            multiple_envs=multiple_envs,
            continuous=env_args.continuous,
            **network_args._asdict(),
        )
    return init_networks_classic(
        env_args.env,
        env_args.env_params,
        network_args.actor_architecture,
        network_args.critic_architecture,
        multiple_envs,
        env_args.continuous,
        action_value=network_args.action_value,
    )


@partial(jax.jit, static_argnames="recurrent")
def get_pi(
    actor_state: TrainState,
    actor_params: FrozenDict,
    obs: np.ndarray,
    hidden: Optional[HiddenState] = None,
    done: Optional[np.ndarray] = None,
    recurrent: bool = False,
) -> tuple[jax.Array, Optional[HiddenState]]:
    """Return the predicted policy"""
    if recurrent:
        jax.debug.callback(check_done_and_hidden, done, hidden)
        return get_pi_rnn(actor_state, actor_params, hidden, obs, done)
    return get_pi_classic(actor_state, actor_params, obs), None
