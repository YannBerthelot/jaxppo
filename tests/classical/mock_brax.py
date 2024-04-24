from typing import Any, Dict, Optional

import jax
import mock
import pytest
from brax import base
from flax import struct
from jax import numpy as jnp
from pytest_mock import mocker

from jaxppo.utils import Environment


@struct.dataclass
class MockPipelineState(base.Base):
    """Environment state for training and inference."""

    x: int


@struct.dataclass
class State(base.Base):
    """Environment state for training and inference."""

    pipeline_state: MockPipelineState
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


def mock_step(state: State, action: jax.Array) -> State:
    # Return mock observations, rewards, and done flag on each step
    mock_pipeline = MockPipelineState(state.pipeline_state.x + 1)
    mock_obs = jnp.ones((4,), dtype=jnp.float32)
    # mock_reward = 1.0
    # mock_done = jax.lax.cond(state.pipeline_state.x > 5, lambda: 1.0, lambda: 0.0)
    mock_reward, mock_done, one = jnp.ones(3)
    mock_metrics = {"mock_metric": one}
    mock_info = {"mock_info": one}
    mock_state = State(
        mock_pipeline, mock_obs, mock_reward, mock_done, mock_metrics, mock_info
    )
    return mock_state


def mock_reset(rng: jax.Array) -> State:
    # Return mock observations on reset
    mock_pipeline = MockPipelineState(0)
    mock_obs = jnp.ones((4,), dtype=jnp.float32)
    mock_reward, mock_done, zero = jnp.zeros(3)
    mock_metrics = {"mock_metric": zero}
    mock_info = {"mock_info": zero}
    mock_state = State(
        mock_pipeline, mock_obs, mock_reward, mock_done, mock_metrics, mock_info
    )
    return mock_state


def test_mock_brax_env(mocker):
    from brax import envs

    env = envs.create("inverted_pendulum")
    mocker.patch.object(env, "step", side_effect=mock_step)
    mocker.patch.object(env, "reset", side_effect=mock_reset)
    state = env.reset(rng=123)
    action = jnp.array([[0.1]], dtype=jnp.float32)
    new_state = env.step(state, action)
    assert new_state.done != 1
    assert isinstance(env, Environment)
