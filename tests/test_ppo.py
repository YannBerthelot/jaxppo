from jax import random
from jaxppo.ppo import get_logprob_and_action
from math import log
import jax.numpy as jnp


def test_get_logprob_and_action():
    logits = jnp.array([-jnp.inf, -jnp.inf, 1.0, -jnp.inf])
    key = random.PRNGKey(42)
    log_prob, action = get_logprob_and_action(key, logits)
    assert action == 2
    assert log_prob == log(1)
