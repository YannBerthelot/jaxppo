import jax
import jax.numpy as jnp
from brax import envs

env = envs.create("halfcheetah")  # Other envs give similar errors.
key = jax.random.PRNGKey(0)
mock_obs = jnp.ones((4,), dtype=jnp.float32)


batch_rng = jax.random.split(jax.random.PRNGKey(0), 64)
batch_state = jax.vmap(jax.jit(env.reset))(batch_rng)
new_state = env.step(env.reset(key), 0.0)
print(batch_state.obs)
