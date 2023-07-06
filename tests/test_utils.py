# pylint: disable = missing-function-docstring, missing-module-docstring
import pytest

from jaxppo.utils import (
    get_env_action_shape,
    get_env_observation_shape,
    get_num_actions,
    get_parameterized_schedule,
    linear_schedule,
    make_envs,
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


def test_make_4_envs():
    num_envs = 4
    envs = make_envs("CartPole-v1", capture_video=False, num_envs=num_envs)
    assert envs.num_envs == num_envs
    assert envs.reset()[0].shape == (num_envs, envs.single_observation_space.shape[0])


def test_make_1_env():
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


def test_get_observation_shape():
    num_envs = 4
    envs = make_envs("CartPole-v1", capture_video=False, num_envs=num_envs)
    observation_space_shape = get_env_observation_shape(envs)
    assert observation_space_shape == (4,)


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
