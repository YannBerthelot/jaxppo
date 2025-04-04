import brax.envs as brax_envs
import gymnax
import jax
import pytest
import wandb
from test_wandb_extraction import (
    create_fake_run_and_get_run_and_id,
    destroy_fake_run,
    get_fake_run,
    setup_wandb_user,
)

from jaxppo.train import UpdateState, init_agent
from jaxppo.video import save_video_to_wandb

setup_wandb_user("yann-berthelot")


@pytest.mark.slow
def test_save_video_gymnax():
    seed = 42
    rng = jax.random.PRNGKey(seed)
    env_id = "CartPole-v1"
    env, env_params = gymnax.make(env_id)
    actor_state, critic_state = (
        init_agent(  # only the env really matters for this test, the rest is placeholder
            key=rng,
            env=env,
            actor_architecture=["8"],
            critic_architecture=["8"],
            num_envs=1,
            anneal_lr=0,
            learning_rate=1e-3,
            num_minibatches=4,
            update_epochs=4,
            num_updates=4,
            max_grad_norm=0.5,
            env_params=env_params,
        )[
            :2
        ]
    )

    update_state = UpdateState(
        actor_state=actor_state,
        critic_state=critic_state,
        traj_batch=None,
        advantages=None,
        targets=None,
        rng=rng,
    )
    _, run_id = create_fake_run_and_get_run_and_id()
    # try:
    save_video_to_wandb(
        env_id=env_id,
        env=env,
        recurrent=False,
        update_state=update_state,
        rng=rng,
        params=env_params,
    )
    wandb.finish()
    fake_run = get_fake_run(run_id)

    files = [file for file in fake_run.files()]

    gif_found = False
    for file in files:
        if ".gif" in str(file.__repr__()) or ".mp4" in str(file.__repr__()):
            gif_found = True
            break
    assert gif_found

    # finally:
    destroy_fake_run(run_id)


@pytest.mark.slow
def test_save_video_brax():
    seed = 42
    rng = jax.random.PRNGKey(seed)
    env_id = "halfcheetah"
    env = brax_envs.create(env_id)
    actor_state, critic_state = (
        init_agent(  # only the env really matters for this test, the rest is placeholder
            key=rng,
            env=env,
            actor_architecture=["8"],
            critic_architecture=["8"],
            num_envs=1,
            anneal_lr=0,
            learning_rate=1e-3,
            num_minibatches=4,
            update_epochs=4,
            num_updates=4,
            max_grad_norm=0.5,
            env_params=None,
        )[
            :2
        ]
    )

    update_state = UpdateState(
        actor_state=actor_state,
        critic_state=critic_state,
        traj_batch=None,
        advantages=None,
        targets=None,
        rng=rng,
    )
    _, run_id = create_fake_run_and_get_run_and_id()
    # try:
    save_video_to_wandb(
        env_id=None,
        env=env,
        recurrent=False,
        update_state=update_state,
        rng=rng,
    )
    wandb.finish()
    fake_run = get_fake_run(run_id)

    files = [file for file in fake_run.files()]

    gif_found = False
    for file in files:
        if ".gif" in str(file.__repr__()) or ".mp4" in str(file.__repr__()):
            gif_found = True
            break
    assert gif_found

    # finally:
    destroy_fake_run(run_id)
