"""Test utils"""
from jaxppo.utils import get_parameterized_schedule, linear_schedule


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
