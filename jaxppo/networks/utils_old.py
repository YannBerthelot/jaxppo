"""Helpfer functions for Networks initialization"""
import functools
import re
from typing import Any, Sequence, TypeAlias, Union, cast, get_args, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from flax.linen.initializers import constant, orthogonal
import pdb

ActivationFunction: TypeAlias = Union[
    jax._src.custom_derivatives.custom_jvp, jaxlib.xla_extension.PjitFunction
]


class HiddenState(NamedTuple):
    h: jnp.ndarray
    c: jnp.ndarray


def check_architecture(actor, num_of_actions):
    if actor and num_of_actions is None:
        raise ValueError("Actor mode is selected but no num_of_actions provided")
    if actor and not isinstance(num_of_actions, int):
        raise ValueError(
            f"Got unexpected num of actions : {num_of_actions} {type(num_of_actions)}"
        )


def reset_hidden_state_where_episode_finished(resets, hidden_state, reset_hidden_state):
    h, c = hidden_state
    reset_h, reset_c = reset_hidden_state
    h = jnp.where(
        resets[:, np.newaxis],
        reset_h,
        h,
    )
    c = jnp.where(
        resets[:, np.newaxis],
        reset_c,
        c,
    )
    return h, c


class ScannedRNN(nn.Module):
    features: int
    idx: int  # idx of the LSTM amongst LSTM layers

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, hidden_state, obs_and_resets):
        """Applies the module."""
        obs, resets = obs_and_resets  # shape : (1,num_envs,obs_shape)
        hidden_state = (
            hidden_state.h,
            hidden_state.c,
        )
        reset_hidden_state = self.initialize_carry(
            jax.random.PRNGKey(0), num_envs=obs.shape[0]
        )
        hidden_state = reset_hidden_state_where_episode_finished(
            resets, hidden_state, reset_hidden_state
        )
        cell = nn.OptimizedLSTMCell(self.features)
        new_hidden_state, embedding = cell(hidden_state, obs)
        obs_and_resets = (embedding, resets)
        h, c = new_hidden_state
        return HiddenState(h=h, c=c), obs_and_resets

    def initialize_carry(self, key, num_envs):
        return nn.OptimizedLSTMCell(self.features, parent=None).initialize_carry(
            key, (num_envs, self.features)
        )


class ScannedDense(nn.Module):
    features: int
    kernel_init: Any  # TODO : do actual typing
    bias_init: Any  # TODO : do actual typing

    @nn.compact
    def __call__(self, hidden_state, obs_and_resets) -> Any:
        obs, resets = obs_and_resets
        embedding = nn.Dense(
            self.features, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(obs)
        obs_and_resets = (embedding, resets)
        return hidden_state, obs_and_resets


class Activation(nn.Module):
    activation_fn: ActivationFunction

    @nn.compact
    def __call__(self, hidden_state, obs_and_resets):
        obs, resets = obs_and_resets
        embedding = self.activation_fn(obs)
        obs_and_resets = (embedding, resets)
        return hidden_state, obs_and_resets


def has_numbers(inputString: str) -> bool:
    """
    Checks wether a string has number inside or not

    Args:
        inputString (str): The string to be checked

    Returns:
        bool: Wether or not the string contains numbers
    """
    return any(char.isdigit() for char in inputString)


def get_LSTM_from_string(string: str, idx: int) -> nn.OptimizedLSTMCell:
    """
    Parse the LSTM architecture to the actual LSTM layer

    Args:
        string (str): The LSTM string representation
        previous_shape (int): The shape of the previous layer

    Returns:
        Tuple[nn.LSTM, int]: The LSTM layer and the number of neurons inside
    """
    LSTM_description = re.search(r"\((.*?)\)", string).group(1)
    nb_neurons = int(LSTM_description)
    return ScannedRNN(features=nb_neurons, idx=idx)


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:  # type: ignore[return]
    """
    Parse string representing activation or jax activation function towards\
        jax activation function
    """
    activation_matching = {"relu": nn.relu, "tanh": nn.tanh}

    match activation:
        case str():
            if activation in activation_matching:
                return Activation(activation_matching[activation])
            else:
                raise ValueError(
                    f"Unrecognized activation name {activation}, acceptable activations"
                    f" names are : {activation_matching.keys()}"
                )
        case activation if isinstance(activation, get_args(ActivationFunction)):
            return Activation(activation)
        case _:
            raise ValueError(f"Unrecognized activation {activation}")


def parse_layer(
    layer: Union[str, ActivationFunction], idx: int
) -> Union[nn.Dense, ActivationFunction]:
    """Parse a layer representation into either a Dense or an activation function"""
    if str(layer).isnumeric():
        return ScannedDense(
            features=int(cast(str, layer)),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
    if "LSTM" in layer:
        return get_LSTM_from_string(layer, idx)
    return parse_activation(activation=layer)


def parse_architecture(
    architecture: Sequence[Union[str, ActivationFunction]],
) -> Sequence[Union[nn.Dense, ActivationFunction]]:
    """Parse a list of string/module architecture into a list of jax modules"""
    layers = []
    i = 0
    for layer in architecture:
        layers.append(parse_layer(layer, i))
        if "LSTM" in layer:
            if i > 0:
                raise ValueError("Only one LSTM layer is allowed")
            i += 1
    return layers
    # return [parse_layer(layer, i) for i, layer in enumerate(architecture)]
