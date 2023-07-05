from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
from jax import Array
from jax.typing import ArrayLike

ActivationFunction = Callable[[ArrayLike], Array]


def parse_activation(activation: str) -> ActivationFunction:
    """Parse string representing activation towards jax activation function"""
    activation_matching = {"relu": nn.relu, "tanh": nn.tanh}
    if activation not in activation_matching.keys():
        raise ValueError(
            f"Unrecognized activation {activation}, acceptable activations are :"
            f" {activation_matching.keys()}"
        )
    return activation_matching[activation]


def parse_layer(layer: str) -> Union[nn.Dense, ActivationFunction]:
    return (
        nn.Dense(int(layer))
        if layer.isnumeric()
        else parse_activation(activation=layer)
    )


def parse_architecture(
    architecture: Sequence[str],
) -> Sequence[Union[nn.Dense, ActivationFunction]]:
    """Parse list of string architecture into a list of jax modules"""
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
