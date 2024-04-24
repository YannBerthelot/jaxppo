# pylint: disable = missing-function-docstring, missing-module-docstring
import pytest

from jaxppo.utils import (
    build_env_from_id,
    get_env_action_shape,
    get_env_observation_shape,
    get_num_actions,
    get_parameterized_schedule,
    linear_schedule,
    make_gymnax_env,
)


def test_linear_schedule():
    """Check that the schedule works properly"""
    assert linear_schedule(initial_learning_rate=1e-3, decay=1e-4, count=5) == 5e-4


def test_get_parameterized_schedule():
    """Check that the schedule works the same when called independently and when \
        called as a partial"""
    schedule = get_parameterized_schedule(
        linear_schedule, initial_learning_rate=1e-3, decay=1e-4
    )
    assert schedule(count=5) == linear_schedule(
        initial_learning_rate=1e-3, decay=1e-4, count=5
    )


def test_get_action_shape_gymnax():
    env, _ = build_env_from_id("CartPole-v1")
    action_space_shape = get_env_action_shape(env)
    assert action_space_shape == ()


def test_get_observation_shape_gymnax():
    env, env_params = build_env_from_id("CartPole-v1")
    observation_space_shape = get_env_observation_shape(env, env_params)
    assert observation_space_shape == (4,)


def test_fail_get_observation_shape_gymnax():
    env, _ = build_env_from_id("CartPole-v1")
    with pytest.raises(ValueError):
        _ = get_env_observation_shape(env)


def test_get_num_actions_gymnax():
    env, env_params = build_env_from_id("CartPole-v1")
    num_actions = get_num_actions(env, env_params)
    assert num_actions == 2


def test_get_num_actions_mjx():
    env, env_params = build_env_from_id("ant")
    num_actions = get_num_actions(env, env_params)
    assert num_actions == 8


def test_get_num_actions_gymnax_continuous():
    env, params, _ = make_gymnax_env("MountainCarContinuous-v0", 42)
    num_actions = get_num_actions(env, params)
    assert num_actions == 1
