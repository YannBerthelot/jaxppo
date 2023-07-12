"""Networks initialization"""
from typing import Callable, Optional, Sequence, Tuple, TypeAlias, Union, cast, get_args

import flax.linen as nn
import gymnasium as gym
import jax
import jaxlib
import numpy as np
import optax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import Array, random
from numpy import ndarray
from optax import GradientTransformation, GradientTransformationExtraArgs
import distrax
from jaxppo.utils import get_num_actions, sample_obs_space

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
    squeeze_value: bool = False
    categorical_output: bool = False

    @property
    def architecture(self) -> nn.Sequential:
        """The architecture of the network as an attribute"""
        if self.actor and self.num_of_actions is None:
            raise ValueError("Actor mode is selected but no num_of_actions provided")
        if self.actor and isinstance(self.num_of_actions, int):
            num_actions = self.num_of_actions
            kernel_init = 0.01
        elif self.actor and not isinstance(self.num_of_actions, int):
            raise ValueError(
                "Got unexpected num of actions :"
                f" {self.num_of_actions} {type(self.num_of_actions)}"
            )
        else:
            num_actions = 1
            kernel_init = 1
        if self.actor:
            return nn.Sequential(
                [
                    *parse_architecture(self.input_architecture),
                    nn.Dense(
                        num_actions,
                        kernel_init=orthogonal(kernel_init),
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
                        num_actions,
                        kernel_init=orthogonal(kernel_init),
                        bias_init=constant(0.0),
                    ),
                ]
            )

    @nn.compact
    def __call__(self, x: Array):
        return (
            self.architecture(x)
            if not (self.squeeze_value)
            else jnp.squeeze(self.architecture(x), axis=-1)
        )


def init_networks(
    env: gym.Env,
    actor_architecture: Sequence[Union[str, ActivationFunction]],
    critic_architecture: Sequence[Union[str, ActivationFunction]],
    squeeze_value: bool = False,
    categorical_output: bool = False,
) -> Tuple[Network, Network]:
    """Create actor and critic adapted to the environment and following the\
          given architectures"""
    num_actions = get_num_actions(env)
    actor = Network(
        input_architecture=actor_architecture,
        actor=True,
        num_of_actions=num_actions,
        categorical_output=categorical_output,
    )
    critic = Network(
        input_architecture=critic_architecture, actor=False, squeeze_value=squeeze_value
    )
    return actor, critic


class AgentParams(struct.PyTreeNode):  # type : ignore
    """Store the actor and critic network parameters"""

    actor_params: FrozenDict
    critic_params: FrozenDict


class AgentState(TrainState):
    """Store the agent training state/parameters"""

    # Setting default values for agent functions to make
    # TrainState work in jitted function
    actor_fn: Callable = struct.field(pytree_node=False)
    critic_fn: Callable = struct.field(pytree_node=False)


def get_adam_tx(
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    max_grad_norm: float = 0.5,
    eps: float = 1e-5,
    clipped=True,
) -> GradientTransformationExtraArgs:
    """Return a clipped adam optimiser with the given parameters"""
    if clipped:
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, eps=eps),
        )
    return optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, eps=eps)


def init_actor_and_critic_state(
    actor_network: Network,
    critic_network: Network,
    actor_key: random.PRNGKeyArray,
    critic_key: random.PRNGKeyArray,
    env: gym.Env,
    tx: Union[GradientTransformationExtraArgs, GradientTransformation],
    env_params=None,
) -> Tuple[TrainState, TrainState]:
    """Returns the proper agent state for the given networks, keys, environment and optimizer (tx)"""
    obs = sample_obs_space(env, env_params)
    actor_params = actor_network.init(actor_key, obs)
    critic_params = critic_network.init(critic_key, obs)
    actor = TrainState.create(params=actor_params, tx=tx, apply_fn=actor_network.apply)
    critic = TrainState.create(
        params=critic_params, tx=tx, apply_fn=critic_network.apply
    )
    return actor, critic


def init_agent_state(
    actor: Network,
    critic: Network,
    actor_key: random.PRNGKeyArray,
    critic_key: random.PRNGKeyArray,
    env: gym.Env,
    tx: Union[GradientTransformationExtraArgs, GradientTransformation],
) -> AgentState:
    """Returns the proper agent state for the given networks, keys, environment and optimizer (tx)"""
    obs = env.reset()[0]
    actor_params = actor.init(actor_key, obs)
    critic_params = critic.init(critic_key, obs)
    agent_params = AgentParams(actor_params=actor_params, critic_params=critic_params)
    return AgentState.create(
        params=agent_params,
        tx=tx,
        # As we have separated actor and critic we don't use apply_fn
        apply_fn=None,
        actor_fn=actor.apply,
        critic_fn=critic.apply,
    )


def predict_value(
    critc_state: AgentState, critic_params: AgentParams, obs: ndarray
) -> Array:
    """Return the predicted value of the given obs with the current critic state"""
    return critc_state.apply_fn(critic_params, obs)  # type: ignore[attr-defined]


def predict_action_logits(
    actor_state: AgentState, actor_params: AgentParams, obs: ndarray
) -> Array:
    """Return the predicted action logits of the given obs with the current actor state"""
    return actor_state.apply_fn(actor_params, obs)  # type: ignore[attr-defined]
