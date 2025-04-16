import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from gymnax.environments.classic_control import CartPole

from jaxppo.evaluate import evaluate
from jaxppo.networks.networks import (
    get_adam_tx,
    init_actor_and_critic_state,
    init_networks,
)
from jaxppo.sac.train_sac_2 import (
    EnvironmentProperties,
    NetworkProperties,
    OptimizerProperties,
)

NUM_EPISODES = 4


@pytest.fixture
def gymnax_env_setup():
    env = CartPole()
    env_params = env.default_params
    env_args = EnvironmentProperties(
        env=env,
        env_params=env_params,
        num_envs=NUM_EPISODES,
        continuous=False,
    )
    return env_args


@pytest.fixture
def brax_env_setup():
    env = create_brax_env("ant", batch_size=NUM_EPISODES)
    env_args = EnvironmentProperties(
        env=env,
        env_params=None,
        num_envs=NUM_EPISODES,
        continuous=True,
    )
    return env_args


@pytest.fixture
def dummy_network_args():
    return NetworkProperties(
        actor_architecture=["64", "64"],
        critic_architecture=["64", "64"],
        lstm_hidden_size=16,
    )


@pytest.fixture
def dummy_optimizer_args():
    return OptimizerProperties(
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )


def get_actor_state(env_args, network_args, optimizer_args, rng, recurrent: bool):
    actor_net, _ = init_networks(env_args, network_args)
    actor_state, _ = init_actor_and_critic_state(
        actor_network=actor_net,
        critic_network=actor_net,  # dummy for testing
        actor_key=rng,
        critic_key=rng,
        actor_tx=get_adam_tx(**optimizer_args._asdict()),
        critic_tx=get_adam_tx(**optimizer_args._asdict()),
        lstm_hidden_size=network_args.lstm_hidden_size if recurrent else None,
        env_args=env_args,
    )
    return actor_state


# --------------------------
#         TESTS
# --------------------------


@pytest.mark.parametrize("recurrent", [False, True])
def test_evaluate_gymnax(
    gymnax_env_setup, dummy_network_args, dummy_optimizer_args, recurrent
):
    key = jax.random.PRNGKey(42)
    env_args = gymnax_env_setup

    if not recurrent:
        dummy_network_args = dummy_network_args._replace(lstm_hidden_size=None)

    actor_state = get_actor_state(
        env_args, dummy_network_args, dummy_optimizer_args, key, recurrent
    )
    rewards, entropy = evaluate(
        env=env_args.env,
        actor_state=actor_state,
        num_episodes=NUM_EPISODES,
        rng=key,
        env_params=env_args.env_params,
        recurrent=recurrent,
        lstm_hidden_size=dummy_network_args.lstm_hidden_size,
    )

    assert isinstance(rewards, jax.Array)
    assert rewards.shape == (NUM_EPISODES,)
    assert jnp.all(rewards >= 0)
    assert entropy.shape == (1,)


@pytest.mark.parametrize("recurrent", [False, True])
def test_evaluate_brax(
    brax_env_setup, dummy_network_args, dummy_optimizer_args, recurrent
):
    key = jax.random.PRNGKey(24)
    env_args = brax_env_setup

    if not recurrent:
        dummy_network_args = dummy_network_args._replace(lstm_hidden_size=None)

    actor_state = get_actor_state(
        env_args, dummy_network_args, dummy_optimizer_args, key, recurrent
    )

    rewards, entropy = evaluate(
        env=env_args.env,
        actor_state=actor_state,
        num_episodes=NUM_EPISODES,
        rng=key,
        env_params=env_args.env_params,
        recurrent=recurrent,
        lstm_hidden_size=dummy_network_args.lstm_hidden_size,
    )

    assert isinstance(rewards, jax.Array)
    assert rewards.shape == (NUM_EPISODES,)
    assert entropy.shape == (1,)
