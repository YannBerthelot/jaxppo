import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from jaxppo.networks.utils import (
    ActivationFunction,
    ScannedDense,
    ScannedRNN,
    get_LSTM_from_string,
    parse_architecture,
    Activation,
)


def check_nn_is_equal(
    nn_1: list[nn.Dense | ActivationFunction],
    nn_2: list[nn.Dense | ActivationFunction],
) -> None:
    """
    Check if layers are equivalent:
    - if it's an activation, check that it's the same type
    - if it's a dense, check it has the same number of neurons
    """
    rng = jax.random.PRNGKey(42)
    for cell_1, cell_2 in zip(nn_1, nn_2):
        assert isinstance(cell_1, type(cell_2))
        if isinstance(cell_1, (nn.linear.Dense, nn.OptimizedLSTMCell, ScannedDense)):
            assert cell_1.features == cell_2.features
            assert jnp.allclose(
                cell_1.kernel_init(rng, (cell_1.features, cell_1.features)),
                cell_2.kernel_init(rng, (cell_1.features, cell_1.features)),
            )
            assert jnp.allclose(
                cell_1.bias_init(rng, (cell_1.features, cell_1.features)),
                cell_2.bias_init(rng, (cell_1.features, cell_1.features)),
            )
        elif isinstance(cell_1, ScannedRNN):
            assert cell_1.features == cell_2.features
            assert cell_1.idx == cell_2.idx
        else:  # activation
            assert cell_1.activation_fn == cell_2.activation_fn
    return True


def test_parse_simple_LSTM():
    LSTM_cell_description = "LSTM(128)"
    expected_LSTM = ScannedRNN(128, 0)
    LSTM_cell = get_LSTM_from_string(LSTM_cell_description, 0)
    assert LSTM_cell.features == expected_LSTM.features


def test_parse_LSTM_network():
    expected_network = [
        ScannedDense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        ),
        Activation(nn.tanh),
        ScannedDense(
            features=32,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        ),
        Activation(nn.relu),
        ScannedRNN(128, 0),
        Activation(nn.relu),
        ScannedDense(
            features=1,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        ),
    ]

    network_description = ["64", "tanh", "32", "relu", "LSTM(128)", "relu", "1"]
    network = parse_architecture(network_description)
    assert check_nn_is_equal(expected_network, network)
