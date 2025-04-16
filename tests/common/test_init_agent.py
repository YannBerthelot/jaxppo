import flashbax as fbx
import jax
import pytest
from gymnax.environments.classic_control.pendulum import Pendulum

from jaxppo.sac.train_sac_2 import (
    AlphaProperties,
    BufferProperties,
    EnvironmentProperties,
    MaybeRecurrentTrainState,
    NetworkProperties,
    OptimizerProperties,
    get_buffer,
    init_agent,
)
from jaxppo.types_rnn import HiddenState  # Assuming this is defined

# Fixtures


@pytest.fixture
def dummy_env_props():
    env = Pendulum()
    num_envs = 1
    env_params = env.default_params
    return EnvironmentProperties(
        env=env,
        env_params=env_params,
        num_envs=num_envs,
        continuous=False,
    )


@pytest.fixture
def dummy_optimizer_props():
    return OptimizerProperties(
        learning_rate=1e-3,
        max_grad_norm=0.5,
    )


@pytest.fixture
def dummy_network_props_non_recurrent():
    return NetworkProperties(
        actor_architecture=("64", "64"),
        critic_architecture=("64", "64"),
        lstm_hidden_size=None,
    )


@pytest.fixture
def dummy_network_props_recurrent():
    return NetworkProperties(
        actor_architecture=("64", "64"),
        critic_architecture=("64", "64"),
        lstm_hidden_size=128,  # Enable LSTM
    )


@pytest.fixture
def dummy_alpha_props():
    return AlphaProperties(alpha_init=0.1, learning_rate=3e-4)


@pytest.fixture
def dummy_buffer():
    dummy_buffer_props = BufferProperties(buffer_size=1000)
    return get_buffer(dummy_buffer_props.buffer_size, dummy_buffer_props.minibatch_size)


def test_init_agent_non_recurrent(
    dummy_env_props,
    dummy_optimizer_props,
    dummy_network_props_non_recurrent,
    dummy_buffer,
    dummy_alpha_props,
):
    key = jax.random.PRNGKey(0)

    (
        rng,
        actor_state,
        critic_state,
        target_critic_state,
        buffer_state,
        alpha_state,
    ) = init_agent(
        key=key,
        env_args=dummy_env_props,
        optimizer_args=dummy_optimizer_props,
        network_args=dummy_network_props_non_recurrent,
        buffer=dummy_buffer,
        alpha_args=dummy_alpha_props,
    )

    assert isinstance(rng, jax.Array)

    assert hasattr(actor_state, "state")
    assert hasattr(critic_state, "__iter__")
    assert isinstance(actor_state, MaybeRecurrentTrainState)

    assert actor_state.state.params is not None
    assert actor_state.hidden_state is None  # Non-recurrent => no hidden state

    for critic_state in (critic_state, target_critic_state):
        for cs in critic_state:
            assert isinstance(cs, MaybeRecurrentTrainState)
            assert cs.state.params is not None
            assert cs.hidden_state is None

    assert isinstance(buffer_state, fbx.flat_buffer.TrajectoryBufferState)


def test_init_agent_recurrent(
    dummy_env_props,
    dummy_optimizer_props,
    dummy_network_props_recurrent,
    dummy_buffer,
    dummy_alpha_props,
):
    key = jax.random.PRNGKey(42)

    (
        rng,
        actor_state,
        critic_state,
        target_critic_state,
        buffer_state,
        alpha_state,
    ) = init_agent(
        key=key,
        env_args=dummy_env_props,
        optimizer_args=dummy_optimizer_props,
        network_args=dummy_network_props_recurrent,
        buffer=dummy_buffer,
        alpha_args=dummy_alpha_props,
    )

    assert isinstance(rng, jax.Array)

    assert hasattr(actor_state, "state")
    assert hasattr(actor_state, "hidden_state")
    assert actor_state.state.params is not None
    assert actor_state.hidden_state is not None
    assert isinstance(actor_state, MaybeRecurrentTrainState)
    assert isinstance(actor_state.hidden_state, HiddenState)

    for critic_state in (critic_state, target_critic_state):
        for cs in critic_state:
            assert isinstance(cs, MaybeRecurrentTrainState)
            assert cs.state.params is not None
            assert cs.hidden_state is not None

    assert isinstance(buffer_state, fbx.flat_buffer.TrajectoryBufferState)


def test_init_agent_jit(
    dummy_env_props,
    dummy_optimizer_props,
    dummy_network_props_non_recurrent,
    dummy_buffer,
    dummy_alpha_props,
):
    from jax import jit

    key = jax.random.PRNGKey(42)

    # Wrap everything in a pure function
    def wrapped_fn(key):
        return init_agent(
            key=key,
            env_args=dummy_env_props,
            optimizer_args=dummy_optimizer_props,
            network_args=dummy_network_props_non_recurrent,
            buffer=dummy_buffer,
            alpha_args=dummy_alpha_props,
        )

    jitted_fn = jit(wrapped_fn)
    out = jitted_fn(key)

    assert out is not None


def test_init_agent_vmap(
    dummy_env_props,
    dummy_optimizer_props,
    dummy_network_props_non_recurrent,
    dummy_buffer,
    dummy_alpha_props,
):
    from jax import vmap

    keys = jax.random.split(jax.random.PRNGKey(123), 4)

    def wrapped_fn(key):
        return init_agent(
            key=key,
            env_args=dummy_env_props,
            optimizer_args=dummy_optimizer_props,
            network_args=dummy_network_props_non_recurrent,
            buffer=dummy_buffer,
            alpha_args=dummy_alpha_props,
        )

    # Warning: your init_agent must be fully pure (no side-effects) for this to work
    vmapped_fn = vmap(wrapped_fn)
    out = vmapped_fn(keys)

    assert out is not None
