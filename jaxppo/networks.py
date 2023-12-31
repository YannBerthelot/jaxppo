"""Networks initialization"""
from typing import Callable, Optional, Sequence, Tuple, TypeAlias, Union, cast, get_args

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams
from jax import Array, random
from numpy import ndarray
from optax import GradientTransformation, GradientTransformationExtraArgs

from jaxppo.utils import get_num_actions

ActivationFunction: TypeAlias = Union[
    jax._src.custom_derivatives.custom_jvp, jaxlib.xla_extension.PjitFunction
]


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:  # type: ignore[return]
    """
    Parse string representing activation or jax activation function towards\
        jax activation function
    """
    activation_matching = {"relu": nn.relu, "tanh": nn.tanh}

    match activation:
        case str():
            if activation in activation_matching:
                return cast(ActivationFunction, activation_matching[activation])
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
    layer: Union[str, ActivationFunction]
) -> Union[nn.Dense, ActivationFunction]:
    """Parse a layer representation into either a Dense or an activation function"""
    if str(layer).isnumeric():
        return nn.Dense(
            int(cast(str, layer)),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
    return parse_activation(activation=layer)


def parse_architecture(
    architecture: Sequence[Union[str, ActivationFunction]],
) -> Sequence[Union[nn.Dense, ActivationFunction]]:
    """Parse a list of string/module architecture into a list of jax modules"""
    return [parse_layer(layer) for layer in architecture]


class Network(nn.Module):
    """
    Generic network that can be actor or critic based on value of "actor" parameter.
    If it's an actor, you have to provide the num_of_actions
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    actor: bool
    num_of_actions: Optional[int] = None
    shared_network: bool = False
    multiple_envs: bool = True

    @property
    def architecture(self) -> nn.Sequential:
        """The architecture of the network as an attribute"""
        if self.actor and self.num_of_actions is None:
            raise ValueError("Actor mode is selected but no num_of_actions provided")
        if self.actor and not isinstance(self.num_of_actions, int):
            raise ValueError(
                "Got unexpected num of actions :"
                f" {self.num_of_actions} {type(self.num_of_actions)}"
            )
        if self.shared_network:
            return nn.Sequential(
                [*parse_architecture(self.input_architecture)],
            )
        else:
            if self.actor:
                return nn.Sequential(
                    [
                        *parse_architecture(self.input_architecture),
                        nn.Dense(
                            self.num_of_actions,
                            kernel_init=orthogonal(0.01),
                            bias_init=constant(0.0),
                        ),
                        distrax.Categorical,
                    ]
                )
            else:
                return nn.Sequential(
                    [
                        *parse_architecture(self.input_architecture),
                        nn.Dense(
                            1,
                            kernel_init=orthogonal(1.0),
                            bias_init=constant(0.0),
                        ),
                    ]
                )

    @nn.compact
    def __call__(self, x: Array):
        if self.shared_network:
            feature_extractor = self.architecture(x)
            pi = nn.Sequential(
                [
                    nn.Dense(
                        self.num_of_actions,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0),
                    ),
                    distrax.Categorical,
                ]
            )(feature_extractor)
            feature_extractor = self.architecture(x)
            val = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            )(feature_extractor)
            return pi, val.squeeze()
        else:
            if self.actor:
                return self.architecture(x)
            return (
                self.architecture(x).squeeze()
                if self.multiple_envs
                else self.architecture(x)
            )


def init_networks(
    env: Environment,
    actor_architecture: Sequence[Union[str, ActivationFunction]],
    critic_architecture: Optional[Sequence[Union[str, ActivationFunction]]] = None,
    shared_network: bool = False,
    multiple_envs: bool = True,
) -> Tuple[Network, Optional[Network]]:
    """Create actor and critic adapted to the environment and following the\
          given architectures"""
    num_actions = get_num_actions(env)
    if shared_network:
        network = Network(
            input_architecture=actor_architecture,
            actor=True,
            num_of_actions=num_actions,
            shared_network=True,
        )
        return network, None
    actor = Network(
        input_architecture=actor_architecture,
        actor=True,
        num_of_actions=num_actions,
    )
    critic = Network(
        input_architecture=critic_architecture, actor=False, multiple_envs=multiple_envs
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
    actor_network: Network,
    actor_key: random.PRNGKeyArray,
    actor_tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    critic_tx: Optional[
        Union[GradientTransformationExtraArgs, GradientTransformation]
    ] = None,
    critic_network: Optional[Network] = None,
    critic_key: Optional[random.PRNGKeyArray] = None,
    shared_network: bool = False,
) -> Tuple[TrainState, Optional[TrainState]]:
    """Returns the proper agent state for the given networks, keys, environment and optimizer (tx)"""
    init_x = jnp.zeros(env.observation_space(env_params).shape)

    if shared_network:
        actor_critic_params = actor_network.init(actor_key, init_x)
        actor_critic = TrainState.create(
            params=actor_critic_params, tx=actor_tx, apply_fn=actor_network.apply
        )
        return actor_critic, None

    if critic_network is None or critic_key is None:
        raise ValueError(
            "Critic network and key should be defined if not using                 "
            " shared network"
        )
    actor_params = actor_network.init(actor_key, init_x)
    critic_params = critic_network.init(critic_key, init_x)
    actor = TrainState.create(
        params=actor_params, tx=actor_tx, apply_fn=actor_network.apply
    )
    critic = TrainState.create(
        params=critic_params, tx=critic_tx, apply_fn=critic_network.apply
    )
    return actor, critic


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
