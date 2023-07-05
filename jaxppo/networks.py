from typing import Optional, Sequence, Union, get_args

import flax.linen as nn
import jax
from jax import Array
import jaxlib
import pdb

ActivationFunction = Union[
    jax._src.custom_derivatives.custom_jvp, jaxlib.xla_extension.PjitFunction
]


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:
    """Parse string representing activation or jax activation function towards \
        jax activation function"""
    activation_matching = {"relu": nn.relu, "tanh": nn.tanh}
    match activation:
        case str() as activation:
            if activation in activation_matching.keys():
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

    # if activation not in activation_matching.keys():
    #     raise ValueError(
    #         f"Unrecognized activation {activation}, acceptable activations are :"
    #         f" {activation_matching.keys()}"
    #     )
    # return activation_matching[activation]


def parse_layer(
    layer: Union[str, ActivationFunction]
) -> Union[nn.Dense, ActivationFunction]:
    """Parse a layer representation into either a Dense or an activation function"""
    return (
        nn.Dense(int(layer))
        if str(layer).isnumeric()
        else parse_activation(activation=layer)
    )


def parse_architecture(
    architecture: Sequence[Union[str, ActivationFunction]],
) -> Sequence[Union[nn.Dense, ActivationFunction]]:
    """Parse a list of string/module architecture into a list of jax modules"""
    return [parse_layer(layer) for layer in architecture]


class Network(nn.Module):
    obs_dim: int
    input_architecture: Sequence[str]
    actor: bool
    num_of_actions: Optional[int] = None

    @property
    def architecture(self):
        return nn.Sequential(
            [
                *parse_architecture(self.input_architecture),
                nn.Dense(self.num_of_actions if self.actor else 1),
            ]
        )

    @nn.compact
    def __call__(self, x: Array):
        return self.architecture(x)
