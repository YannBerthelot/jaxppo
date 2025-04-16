"""Networks initialization"""

from typing import Callable, Optional, Sequence, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams
from jax import Array
from numpy import ndarray
from optax import GradientTransformation, GradientTransformationExtraArgs

from jaxppo.networks.utils import (
    ActivationFunction,
    ScannedRNN,
    check_architecture,
    parse_architecture,
)
from jaxppo.types_rnn import HiddenState
from jaxppo.utils import get_num_actions, get_observation_space_shape


class NetworkRNN(nn.Module):
    """
    Generic network that can be actor or critic based on value of "actor" parameter.
    If it's an actor, you have to provide the num_of_actions
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    actor: bool
    num_of_actions: Optional[int] = None
    multiple_envs: bool = True
    lstm_hidden_size: int = 64
    continuous: bool = False
    action_value: bool = False

    @property
    def extractor_architecture(self) -> nn.Sequential:
        """The architecture of the network as an attribute"""
        check_architecture(self.actor, self.num_of_actions)
        return parse_architecture(self.input_architecture)

    @nn.compact
    def __call__(
        self,
        hiddens: list[Array],
        actor_critic_in: tuple[Array, Array],
        action: Optional[Array] = None,
    ) -> tuple[list[HiddenState], Array]:
        obs, dones = actor_critic_in

        if self.action_value:
            embedding = jnp.concatenate([obs, action], axis=-1)
        else:
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
            if self.continuous:
                actor_logtstd = self.param(
                    "log_std",
                    nn.initializers.zeros,
                    self.num_of_actions,
                )
                return distrax.Normal(logits, jnp.exp(actor_logtstd)), extractor_hiddens
            return distrax.Categorical(logits=logits), extractor_hiddens
        val = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            embedding
        )
        return (
            jnp.squeeze(val, axis=-1) if self.multiple_envs else val
        ), extractor_hiddens


def init_hidden_state(
    lstm_hidden_size: int,
    num_envs: int,
    rng: jax.random.PRNGKey,
) -> HiddenState:
    """Initialize the hidden state for the recurrent layer of the network."""
    # rng, _rng = jax.random.split(rng)
    return ScannedRNN(lstm_hidden_size).initialize_carry(rng, num_envs)


def init_networks(
    env: Environment,
    params: EnvParams,
    actor_architecture: Sequence[Union[str, ActivationFunction]],
    critic_architecture: Sequence[Union[str, ActivationFunction]],
    multiple_envs: bool = True,
    lstm_hidden_size: int = 64,
    continuous: bool = False,
    action_value: bool = False,
) -> Tuple[NetworkRNN, NetworkRNN]:
    """Create actor and critic adapted to the environment and following the\
          given architectures"""
    actor = NetworkRNN(
        input_architecture=actor_architecture,
        actor=True,
        num_of_actions=get_num_actions(env, params),
        lstm_hidden_size=lstm_hidden_size,
        continuous=continuous,
    )
    critic = NetworkRNN(
        input_architecture=critic_architecture,
        actor=False,
        multiple_envs=multiple_envs,
        lstm_hidden_size=lstm_hidden_size,
        action_value=action_value,
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


def get_model_and_state(
    network: Union[NetworkRNN, Sequence[NetworkRNN]],
    key: jax.random.PRNGKey,
    lstm_hidden_size: int,
    num_envs: int,
    tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    init_x: jax.Array,
):
    if isinstance(network, NetworkRNN):
        network = [network]
    models, hidden_states = [], []
    for _network in network:
        sub_key, key = jax.random.split(key)
        _init_hidden_state = init_hidden_state(lstm_hidden_size, num_envs, sub_key)
        params = _network.init(sub_key, _init_hidden_state, init_x)
        model = TrainState.create(params=params, tx=tx, apply_fn=_network.apply)
        models.append(model)
        hidden_states.append(_init_hidden_state)
    models = tuple(models)
    if len(models) == 1:
        return models[0], hidden_states[0]
    return models, hidden_states


def init_actor_and_critic_state(
    env: Environment,
    env_params: EnvParams,
    rng: jax.random.PRNGKey,
    num_envs: int,
    actor_network: NetworkRNN,
    actor_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_network: NetworkRNN,
    lstm_hidden_size: int = 64,
    action_value: bool = False,
) -> Tuple[tuple[TrainState, TrainState], tuple[HiddenState, HiddenState]]:
    """Returns the proper agent state for the given networks, keys, environment and optimizer (tx)"""
    init_x = (
        jnp.zeros((1, num_envs, get_observation_space_shape(env, env_params))),
        jnp.zeros((1, num_envs)),
    )
    rng, actor_key, critic_key = jax.random.split(rng, num=3)

    actor, init_hidden_state_actor = get_model_and_state(
        actor_network, actor_key, lstm_hidden_size, num_envs, actor_tx, init_x
    )
    critic, init_hidden_state_critic = get_model_and_state(
        critic_network, critic_key, lstm_hidden_size, num_envs, critic_tx, init_x
    )

    return (actor, critic), (init_hidden_state_actor, init_hidden_state_critic)


def predict_probs_and_value(
    actor_critic_state: TrainState, actor_critic_params: FrozenDict, obs: ndarray
) -> Tuple[Array, Array]:
    """Return the predicted probs and value of the given obs with the current \
        actor critic state"""
    pi, val = actor_critic_state.apply_fn(actor_critic_params, obs)
    return pi.probs, val


def predict_value_and_hidden(
    critic_state: TrainState,
    critic_params: FrozenDict,
    hidden: HiddenState,
    obs: ndarray,
    done: ndarray,
    action: Optional[ndarray] = None,
) -> tuple[Array, HiddenState]:
    """Return the predicted value of the given obs with the current critic state"""
    return critic_state.apply_fn(critic_params, hidden, (obs, done), action)  # type: ignore[attr-defined]


def predict_probs(
    actor_state: TrainState,
    actor_params: FrozenDict,
    hidden: HiddenState,
    obs: ndarray,
    done: ndarray,
    action: Optional[ndarray] = None,
) -> Array:
    """Return the predicted action logits of the given obs with the current actor state"""
    return actor_state.apply_fn(actor_params, hidden, (obs, done), action)[0].probs  # type: ignore[attr-defined]


def get_pi(
    actor_state: TrainState,
    actor_params: FrozenDict,
    hidden: HiddenState,
    obs: ndarray,
    done: ndarray,
) -> Array:
    """Return the predicted action logits of the given obs with the current actor state"""
    return actor_state.apply_fn(actor_params, hidden, (obs, done))  # type: ignore[attr-defined]
