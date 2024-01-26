from functools import partial
from typing import Callable, Optional, Sequence, TypeAlias, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
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


@partial(jax.jit, static_argnames="recurrent")
def predict_action_and_prob(
    actor_state: TrainState,
    actor_params: FrozenDict,
    obs: np.ndarray,
    key: jnp.array,
    hidden: Optional[HiddenState] = None,
    done: Optional[np.ndarray] = None,
    recurrent: bool = False,
) -> jax.Array:
    """Return the predicted action logits of the given obs with the current actor state"""
    if recurrent:
        jax.debug.callback(check_done_and_hidden, done, hidden)
        pass
        # return predict_prob_rnn(actor_state, actor_params, hidden, obs, done, key)
    return predict_prob_classic(actor_state, actor_params, obs, key)


@partial(jax.jit, static_argnames="recurrent")
def predict_value(
    critic_state: TrainState,
    critic_params: FrozenDict,
    obs: np.ndarray,
    hidden: Optional[HiddenState] = None,
    done: Optional[np.ndarray] = None,
    recurrent: bool = False,
) -> tuple[jax.Array, Optional[HiddenState]]:
    """Return the predicted value of the given obs with the current critic state"""
    if recurrent:
        jax.debug.callback(check_done_and_hidden, done, hidden)
        return predict_value_and_hidden(critic_state, critic_params, hidden, obs, done)
    return predict_value_classic(critic_state, critic_params, obs), None


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
    env: Environment,
    env_params: EnvParams,
    actor_network: Union[NetworkClassic, NetworkRNN],
    actor_key: jax.Array,
    actor_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_network: Union[NetworkClassic, NetworkRNN],
    critic_key: Optional[jax.Array] = None,
    num_envs: Optional[int] = None,
    lstm_hidden_size: Optional[int] = None,
) -> tuple[
    tuple[TrainState, TrainState], tuple[Optional[HiddenState], Optional[HiddenState]]
]:
    """Initialize the actor and critic train state (and hidden states if needed) given\
          their networks."""
    if lstm_hidden_size is not None:
        if num_envs is None or lstm_hidden_size is None:
            raise ValueError(
                "Num envs and lstm hidden size should be defined when using rnn, got"
                f" {num_envs=} and {lstm_hidden_size=}"
            )
        if not (
            isinstance(actor_network, NetworkRNN)
            and isinstance(critic_network, NetworkRNN)
        ):
            raise ValueError(
                "Networks should be recurrent when using recurrent mode, got"
                f" {actor_network=} and {critic_network=}"
            )
        return init_actor_and_critic_state_rnn(
            env,
            env_params,
            actor_key,
            num_envs,
            actor_network,
            actor_tx,
            critic_tx,
            critic_network,
            lstm_hidden_size,
        )
    return (
        init_actor_and_critic_state_classic(
            env,
            env_params,
            actor_network,
            actor_key,
            actor_tx,
            critic_tx,
            critic_network,
            critic_key,
        ),
        (None, None),
    )


def init_networks(
    env: Environment,
    actor_architecture: Sequence[Union[str, ActivationFunction]],
    critic_architecture: Sequence[Union[str, ActivationFunction]],
    multiple_envs: bool,
    lstm_hidden_size: Optional[int] = None,
    continuous: bool = False,
) -> Union[tuple[NetworkClassic, NetworkClassic], tuple[NetworkRNN, NetworkRNN]]:
    if lstm_hidden_size is not None:
        return init_networks_rnn(
            env,
            actor_architecture,
            critic_architecture,
            multiple_envs,
            lstm_hidden_size,
            continuous,
        )
    return init_networks_classic(
        env, actor_architecture, critic_architecture, multiple_envs, continuous
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
