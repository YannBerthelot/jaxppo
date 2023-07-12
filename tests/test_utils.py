# pylint: disable = missing-function-docstring, missing-module-docstring
import gymnasium as gym
import gymnax
import pytest
from gymnax.environments.environment import Environment
from jax import Array

from jaxppo.utils import (
    get_env_action_shape,
    get_env_observation_shape,
    get_num_actions,
    get_parameterized_schedule,
    linear_schedule,
    make_envs,
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


def test_make_envs_from_env():
    num_envs = 4
    env = gym.make("CartPole-v1")
    envs = make_envs(env, capture_video=False, num_envs=num_envs)
    assert envs.num_envs == num_envs
    assert envs.reset()[0].shape == (num_envs, envs.single_observation_space.shape[0])


def test_make_gymnax_env():
    env, env_params, keys = make_gymnax_env(env_id="CartPole-v1", seed=42)
    assert isinstance(env, Environment)
    assert isinstance(
        env_params, gymnax.environments.classic_control.cartpole.EnvParams
    )

    assert isinstance(keys[0], Array)


def test_make_4_envs_from_id():
    num_envs = 4
    envs = make_envs("CartPole-v1", capture_video=False, num_envs=num_envs)
    assert envs.num_envs == num_envs
    assert envs.reset()[0].shape == (num_envs, envs.single_observation_space.shape[0])


def test_make_1_env_from_id():
    num_envs = 1
    envs = make_envs("CartPole-v1", capture_video=False, num_envs=num_envs)
    assert envs.num_envs == num_envs
    assert envs.reset()[0].shape == (num_envs, envs.single_observation_space.shape[0])


def test_get_action_shape():
    num_envs = 4
    envs = make_envs("CartPole-v1", capture_video=False, num_envs=num_envs)
    action_space_shape = get_env_action_shape(envs)
    assert action_space_shape == (1,)
    with pytest.raises(AssertionError):
        envs = make_envs(
            "MountainCarContinuous-v0", capture_video=False, num_envs=num_envs
        )
        action_space_shape = get_env_action_shape(envs)
        assert action_space_shape == (1,)


def test_get_action_shape_gymnax():
    env, _, _ = make_gymnax_env("CartPole-v1", 42)
    action_space_shape = get_env_action_shape(env)
    assert action_space_shape == ()
    with pytest.raises(AssertionError):
        env, _, _ = make_gymnax_env("MountainCarContinuous-v0", 42)
        action_space_shape = get_env_action_shape(env)
        assert action_space_shape == ()


def test_get_observation_shape():
    num_envs = 4
    envs = make_envs("CartPole-v1", capture_video=False, num_envs=num_envs)
    observation_space_shape = get_env_observation_shape(envs)
    assert observation_space_shape == (4,)


def test_get_observation_shape_gymnax():
    env, env_state, _ = make_gymnax_env("CartPole-v1", 42)
    observation_space_shape = get_env_observation_shape(env, env_state)
    assert observation_space_shape == (4,)


def test_fail_get_observation_shape_gymnax():
    env, _, _ = make_gymnax_env("CartPole-v1", 42)
    with pytest.raises(ValueError):
        _ = get_env_observation_shape(env)


def test_get_num_actions():
    num_envs = 4
    envs = make_envs("CartPole-v1", capture_video=False, num_envs=num_envs)
    num_actions = get_num_actions(envs)
    assert num_actions == 2
    with pytest.raises(AssertionError):
        envs = make_envs(
            "MountainCarContinuous-v0", capture_video=False, num_envs=num_envs
        )
        num_actions = get_num_actions(envs)
        assert num_actions == 1


def test_get_num_actions_gymnax():
    env, _, _ = make_gymnax_env("CartPole-v1", 42)
    num_actions = get_num_actions(env)
    assert num_actions == 2
