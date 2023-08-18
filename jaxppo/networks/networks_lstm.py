"""Networks initialization"""
from typing import Callable, Optional, Sequence, Tuple, Union, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams
from jax import Array, random
from numpy import ndarray
from optax import GradientTransformation, GradientTransformationExtraArgs
import functools
from jaxppo.networks.utils import (
    ActivationFunction,
    ScannedRNN,
    check_architecture,
    parse_architecture,
)
from jaxppo.utils import get_num_actions


class NetworkLSTM(nn.Module):
    """
    Generic network that can be actor or critic based on value of "actor" parameter.
    If it's an actor, you have to provide the num_of_actions
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    actor: bool
    num_of_actions: Optional[int] = None
    shared_network: bool = False
    multiple_envs: bool = True
    lstm_hidden_size: int = 64

    @property
    def extractor_architecture(self) -> nn.Sequential:
        """The architecture of the network as an attribute"""
        check_architecture(self.actor, self.num_of_actions)
        return parse_architecture(self.input_architecture)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, hiddens: list[Array], actor_critic_in: tuple[Array, Array]
    ) -> tuple[list[Array], Array]:
        obs, dones = actor_critic_in
        # embedding = nn.Sequential(self.extractor_architecture)(obs)
        embedding = obs
        extractor_hiddens, embedding = ScannedRNN(self.lstm_hidden_size)(
            hiddens, (embedding, dones)
        )
        embedding = nn.Sequential(self.extractor_architecture)(embedding)
        if self.actor:
            logits = nn.Dense(
                self.num_of_actions,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(embedding)
            return extractor_hiddens, distrax.Categorical(logits=logits)
        val = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            embedding
        )
        return extractor_hiddens, (
            jnp.squeeze(val, axis=-1) if self.multiple_envs else val
        )


def init_hidden_state(
    layer,
    num_envs: int,
    rng: jax.random.PRNGKey,
) -> tuple:
    """Initialize the hidden state for every LSTM layer in the network."""
    rng, _rng = jax.random.split(rng)
    return layer.initialize_carry(_rng, num_envs=num_envs)


def init_networks(
    env: Environment,
    actor_architecture: Sequence[Union[str, ActivationFunction]],
    critic_architecture: Optional[Sequence[Union[str, ActivationFunction]]] = None,
    shared_network: bool = False,
    multiple_envs: bool = True,
    lstm_hidden_size: int = 64,
) -> Tuple[NetworkLSTM, Optional[NetworkLSTM]]:
    """Create actor and critic adapted to the environment and following the\
          given architectures"""
    actor = NetworkLSTM(
        input_architecture=actor_architecture,
        actor=True,
        num_of_actions=get_num_actions(env),
        lstm_hidden_size=lstm_hidden_size,
    )
    critic = NetworkLSTM(
        input_architecture=critic_architecture,
        actor=False,
        multiple_envs=multiple_envs,
        lstm_hidden_size=lstm_hidden_size,
    )
    return actor, critic


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
    rng: jax.random.PRNGKey,
    num_envs: int,
    actor_network: NetworkLSTM,
    actor_key: random.PRNGKeyArray,
    actor_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_tx: Optional[
        Union[GradientTransformationExtraArgs, GradientTransformation]
    ] = None,
    critic_network: Optional[NetworkLSTM] = None,
    critic_key: Optional[random.PRNGKeyArray] = None,
    shared_network: bool = False,
    lstm_hidden_size: int = 64,
) -> Tuple[TrainState, Optional[TrainState]]:
    """Returns the proper agent state for the given networks, keys, environment and optimizer (tx)"""
    init_x = (
        jnp.zeros((1, num_envs, *env.observation_space(env_params).shape)),
        jnp.zeros((1, num_envs)),
    )
    init_hstate_actor = init_hidden_state(ScannedRNN(lstm_hidden_size), num_envs, rng)
    init_hstate_critic = init_hidden_state(ScannedRNN(lstm_hidden_size), num_envs, rng)
    if critic_network is None or critic_key is None:
        raise ValueError(
            "Critic network and key should be defined if not using                 "
            " shared network"
        )
    actor_params = actor_network.init(actor_key, init_hstate_actor, init_x)
    critic_params = critic_network.init(critic_key, init_hstate_critic, init_x)
    actor = TrainState.create(
        params=actor_params, tx=actor_tx, apply_fn=actor_network.apply
    )
    critic = TrainState.create(
        params=critic_params, tx=critic_tx, apply_fn=critic_network.apply
    )
    return actor, critic, init_hstate_actor, init_hstate_critic


def predict_probs_and_value(
    actor_critic_state: TrainState, actor_critic_params: FrozenDict, obs: ndarray
) -> Tuple[Array, Array]:
    """Return the predicted probs and value of the given obs with the current \
        actor critic state"""
    pi, val = actor_critic_state.apply_fn(actor_critic_params, obs)
    return pi.probs, val


def predict_value(
    critic_state: TrainState, critic_params: FrozenDict, obs: ndarray
) -> Array:
    """Return the predicted value of the given obs with the current critic state"""
    return critic_state.apply_fn(critic_params, obs)  # type: ignore[attr-defined]


def predict_probs(
    actor_state: TrainState, actor_params: FrozenDict, obs: ndarray
) -> Array:
    """Return the predicted action logits of the given obs with the current actor state"""
    return actor_state.apply_fn(actor_params, obs).probs  # type: ignore[attr-defined]
