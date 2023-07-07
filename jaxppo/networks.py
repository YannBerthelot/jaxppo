"""Networks initialization"""
from typing import Callable, Optional, Sequence, Tuple, TypeAlias, Union, cast, get_args

import flax.linen as nn
import gymnasium as gym
import jax
import jaxlib
import optax
from flax import struct
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
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
        return nn.Dense(int(cast(str, layer)))
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

    @property
    def architecture(self) -> nn.Sequential:
        """The architecture of the network as an attribute"""
        if self.actor and self.num_of_actions is None:
            raise ValueError("Actor mode is selected but no num_of_actions provided")
        if self.actor and isinstance(self.num_of_actions, int):
            num_actions = self.num_of_actions
        elif self.actor and not isinstance(self.num_of_actions, int):
            raise ValueError(
                "Got unexpected num of actions :"
                f" {self.num_of_actions} {type(self.num_of_actions)}"
            )
        else:
            num_actions = 1
        return nn.Sequential(
            [
                *parse_architecture(self.input_architecture),
                nn.Dense(num_actions),
            ]
        )

    @nn.compact
    def __call__(self, x: Array):
        return self.architecture(x)


def init_networks(
    env: gym.Env,
    actor_architecture: Sequence[Union[str, ActivationFunction]],
    critic_architecture: Sequence[Union[str, ActivationFunction]],
) -> Tuple[Network, Network]:
    """Create actor and critic adapted to the environment and following the\
          given architectures"""
    num_actions = get_num_actions(env)
    actor = Network(
        input_architecture=actor_architecture,
        actor=True,
        num_of_actions=num_actions,
    )
    critic = Network(
        input_architecture=critic_architecture,
        actor=False,
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
) -> GradientTransformationExtraArgs:
    """Return a clipped adam optimiser with the given parameters"""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate, eps=eps),
    )


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
    actor_params = freeze(actor.init(actor_key, obs))
    critic_params = freeze(critic.init(critic_key, obs))
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
    agent_state: AgentState, agent_params: AgentParams, obs: ndarray
) -> Array:
    """Return the predicted value of the given obs with the current critic state"""
    return agent_state.critic_fn(agent_params.critic_params, obs)  # type: ignore[attr-defined]


def predict_action_logits(
    agent_state: AgentState, agent_params: AgentParams, obs: ndarray
) -> Array:
    """Return the predicted action logits of the given obs with the current actor state"""
    return agent_state.actor_fn(agent_params.actor_params, obs)  # type: ignore[attr-defined]
