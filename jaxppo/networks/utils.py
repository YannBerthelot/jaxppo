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


def check_architecture(actor, num_of_actions):
    if actor and num_of_actions is None:
        raise ValueError("Actor mode is selected but no num_of_actions provided")
    if actor and not isinstance(num_of_actions, int):
        raise ValueError(
            f"Got unexpected num of actions : {num_of_actions} {type(num_of_actions)}"
        )


def reset_hidden_state_where_episode_finished(resets, hidden_state, reset_hidden_state):
    # h, c = hidden_state
    # reset_h, reset_c = reset_hidden_state
    return jnp.where(
        resets[:, np.newaxis],
        reset_hidden_state,
        hidden_state,
    )
    # c = jnp.where(
    #     resets[:, np.newaxis],
    #     reset_c,
    #     c,
    # )
    # return h, c


class ScannedRNN(nn.Module):
    features: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, hidden_state, obs_and_resets):
        """Applies the module."""
        obs, resets = obs_and_resets

        reset_hidden_state = self.initialize_carry(
            jax.random.PRNGKey(0), num_envs=obs.shape[0]
        )

        hidden_state = reset_hidden_state_where_episode_finished(
            resets, hidden_state, reset_hidden_state
        )

        cell = nn.GRUCell(self.features)
        new_hidden_state, embedding = cell(hidden_state, obs)
        return new_hidden_state, embedding

    def initialize_carry(self, key, num_envs):
        return nn.GRUCell(self.features, parent=None).initialize_carry(
            key, (num_envs, self.features)
        )


def has_numbers(inputString: str) -> bool:
    """
    Checks wether a string has number inside or not

    Args:
        inputString (str): The string to be checked

    Returns:
        bool: Wether or not the string contains numbers
    """
    return any(char.isdigit() for char in inputString)


def get_LSTM_from_string(string: str) -> nn.OptimizedLSTMCell:
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
    return ScannedRNN(features=nb_neurons)


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:  # type: ignore[return]
    """
    Parse string representing activation or jax activation function towards\
        jax activation function
    """
    activation_matching = {"relu": nn.relu, "tanh": nn.tanh}

    match activation:
        case str():
            if activation in activation_matching:
                return activation_matching[activation]
            else:
                raise ValueError(
                    f"Unrecognized activation name {activation}, acceptable activations"
                    f" names are : {activation_matching.keys()}"
                )
        case activation if isinstance(activation, get_args(ActivationFunction)):
            return activation
        case _:
            raise ValueError(f"Unrecognized activation {activation}")


def parse_layer(
    layer: Union[str, ActivationFunction], idx: int
) -> Union[nn.Dense, ActivationFunction]:
    """Parse a layer representation into either a Dense or an activation function"""
    if str(layer).isnumeric():
        return nn.Dense(
            features=int(cast(str, layer)),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
    if "LSTM" in layer:
        return get_LSTM_from_string(layer)
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
