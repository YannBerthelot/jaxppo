"""Test ppo"""
import jax

from jaxppo.ppo_pure_rl import make_train


def test_ppo_train():
    """Test that the ppo train function doesn't fail"""
    seed = 42
    key = jax.random.PRNGKey(seed)
    num_envs = 8
    total_timesteps = int(1e6)
    num_steps = 128
    learning_rate = 2.5e-4
    env_id = "CartPole-v1"

    train_jit = jax.jit(
        make_train(
            total_timesteps=total_timesteps,
            num_steps=num_steps,
            num_envs=num_envs,
            env_id=env_id,
            learning_rate=learning_rate,
            actor_architecture=["64", "tanh", "64", "tanh"],
            critic_architecture=["64", "tanh", "64", "tanh"],
        )
    )
    train_jit(key)
