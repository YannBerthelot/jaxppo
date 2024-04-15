"""Tests for wandb extractions of multiple runs from a single merged-run"""

import os
import shutil
from typing import Any

import jax
import wandb

from jaxppo.wandb_extraction import split_run

WandbRun = Any


def log_smth(run: WandbRun, val: Any) -> None:
    """Log  "fake" values"""
    run.log({"value": val, "other_val": val})


PROJECT = "Unit tests"
RUN = "Fake run"
CONFIG = {"A": 1}


def setup_wandb_user(username):
    """Create an env variable to setup the user."""
    os.system(f"export WANDB_ENTITY={username}")


setup_wandb_user("yann-berthelot")


def get_fake_run(run_id):
    """Fetch the fake run from W&B API"""
    api = wandb.Api()
    return api.run(f"{os.environ['WANDB_ENTITY']}/{PROJECT}/{run_id}")


def create_fake_run_and_get_run_and_id(folder: str = "wandb_fake"):
    """Create a fake run and upload it to W&B"""
    run_id = wandb.util.generate_id()
    if folder != "wandb":
        os.makedirs(folder, exist_ok=True)
    run = wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=PROJECT,
        name=RUN,
        config=CONFIG,
        mode="online",
        id=run_id,
        dir=folder,
    )
    return run, run_id


def create_fake_run(fake_folder: str = "wandb_fake"):
    """Create a fake run and upload it to W&B"""
    run, run_id = create_fake_run_and_get_run_and_id(fake_folder)

    def run_loop(value) -> None:
        """Log value onto W&B for 10 timesteps"""
        for _ in range(10):
            jax.debug.callback(log_smth, run, value)
        wandb.finish()

    values = jax.numpy.array([1, 2])
    jax.vmap(run_loop, in_axes=0)(values)
    return run_id


def destroy_fake_run(run_id, folder: str = "wandb_fake") -> None:
    """Remove the fake run from W&B using its id"""
    api = wandb.Api()
    run = api.run(f"{os.environ['WANDB_ENTITY']}/{PROJECT}/{run_id}")
    run.delete()
    shutil.rmtree(folder)


def test_split_run():
    """Checks that the merged-run is split into the expected individual runs."""
    fake_folder = "wandb_fake"
    run_id = create_fake_run(fake_folder)
    try:
        run = get_fake_run(run_id)
        runs, config, name = split_run(run, num_splits=2)
    finally:
        destroy_fake_run(run_id, fake_folder)
    (run_1, run_2) = runs
    for key in ["value", "other_val"]:
        assert [val[key] for val in run_1] == [1 for _ in range(10)]
        assert [val[key] for val in run_2] == [2 for _ in range(10)]
    assert config == CONFIG
    assert name == RUN
