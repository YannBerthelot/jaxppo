"""
Extract multiple runs from a merged-wandb run due to lack of multithreading.
See: https://community.wandb.ai/t/how-to-split-runs-because-of-lack-of-multithreading-given-a-metric/4856
"""

import glob
import json
import os
from typing import Any, Generator, Optional

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from tqdm import tqdm

WandbRun = Any
os.environ["WANDB_ENTITY"] = "yann-berthelot"


def get_run(run_number: int, user_name: str, project_name: str):
    """Extracts the i-th run of the user/project from the W&B API, 0 being last."""
    api = wandb.Api()
    runs = api.runs(f"{user_name}/{project_name}")
    return runs[run_number]


def split_run(run: WandbRun, num_splits: int) -> tuple[list, dict, str]:
    """
    Split the last wandb run.

    Args:
        user_name (str): The user for which to select the run
        project_name (str): The wandb project run
        num_splits (int): The number of runs to split into

    Returns:
        tuple[list, dict, str]: The split runs, the config, and the original run name
    """

    history, config, name = get_history_config_and_name_from_run(run)
    return (
        [list(history)[i::num_splits] for i in range(num_splits)],
        config,
        name,
    )


def get_config_from_run(run: WandbRun) -> dict:
    """Extract config from a wandb run"""
    return {key: value["value"] for key, value in json.loads(run.json_config).items()}


def get_history_from_run(run: WandbRun, page_size: int = int(1e6)) -> Generator:
    """
    Extract history from wandb run

    Args:
        run (wandb.run): The wandb run to consider
        page_size (int, optional): How much steps to process at a time,\
              the higher the better perf. Defaults to int(1e6).

    Returns:
        Generator: The history of the run

    """
    return run.scan_history(page_size=page_size)


def get_name_from_run(run: WandbRun) -> str:
    """Extract name from wandb run"""
    return run.name


def get_history_config_and_name_from_run(run: WandbRun) -> tuple[Generator, dict, str]:
    """Extract history, config, and name from last wandb run with the given\
          user and project"""
    return get_history_from_run(run), get_config_from_run(run), get_name_from_run(run)


def upload_runs(
    project_name: str,
    run_name: str,
    runs: list,
    config: dict,
    folder: Optional[str] = None,
) -> None:
    """Build the runs locally then upload the collected runs"""
    for i, run in tqdm(enumerate(runs)):
        run_config = config
        run_config["seed"] = i
        wandb_run = wandb.init(
            project=project_name,
            name=f"{run_name}_{i}",
            config=config,
            reinit=True,
            mode="offline",
            dir=folder,
        )
        # Build
        for step in tqdm(run, leave=False):
            wandb_run.log(step)
        wandb_run.finish()

        # Upload
        BASE_FOLDER = "./wandb"
        file = glob.glob(f"{BASE_FOLDER}/*{wandb_run.id}")[0]
        os.system(f"wandb sync --clean-force {file} ")


def save_split_data_to_csv(split_data, config, run_name):
    folder_name = "csv_data"
    os.makedirs(os.path.join(folder_name, run_name), exist_ok=True)
    with open(os.path.join(folder_name, run_name, "config.json"), "w") as outfile:
        json.dump(config, outfile)
    df = pd.DataFrame.from_dict(split_data[0])
    df["seed"] = 0
    for i, run in enumerate(split_data[1:]):
        run_df = pd.DataFrame.from_dict(run)
        run_df["seed"] = i + 1
        df = pd.concat((df, run_df))
    df.to_csv(os.path.join(folder_name, run_name, f"{run_name}.csv"))


def split_runs_and_upload_them(
    runs: list[int],
    user_name: str,
    project_name_in: str,
    num_split: int,
    project_name_out: Optional[str] = None,
    folder: str = "wandb",
) -> None:
    """
    Split the defined wandb run of the project/user into runs into the project_name_out.
    First run is the 1st element, then the 1 + num_split-th element etc ...
    Second run is the 2nd element, then the 2 + num_split-th element etc ...

    If no project_out is given, take the initial project.

    Args:
        user_name (str): The user to consider.
        project_name_in (str): The project to take the last run in.
        num_split (int): The number of splits to make.
        project_name_out (Optional[str], optional): _description_. Defaults to None.
    """
    print("Splitting run")
    for run_number in tqdm(runs):
        run = get_run(
            run_number, user_name, project_name_in
        )  # TODO :Be able to work with other runs? by name ?
        runs, config, name = split_run(run, num_split)
        save_split_data_to_csv(runs, config, name)
        if project_name_out is None:
            project_name_out = project_name_in
        print("Runs split. Uploading runs.")
        upload_runs(project_name_out, name, runs, config, folder=folder)
        print("Uploading done.")


def concat_multiple_files():
    raise NotImplementedError


def plot_area(top, bottom, middle):
    raise NotImplementedError


def get_top_bottom_and_middle_from_column():
    raise NotImplementedError


def plot_and_save_agregated_metrics():
    raise NotImplementedError


if __name__ == "__main__":
    runs = list(range(11))
    # split_runs_and_upload_them(
    #     runs,
    #     os.environ["WANDB_ENTITY"],
    #     project_name_in="Benchmark delay merged",
    #     project_name_out="Benchmark delay merged out",
    #     num_split=100,
    # )

    df = pd.read_csv(
        (
            "/Users/yann/Documents/jaxppo/csv_data/jax delayed delay=10/jax delayed"
            " delay=10.csv"
        ),
        index_col=0,
    )

    aggregated_df = df.reset_index().groupby("index")

    def get_aggregation_from_df(df, aggregator):
        if aggregator == "mean":
            return df.mean()
        raise NotImplementedError

    mean_df = get_aggregation_from_df(aggregated_df, "mean")

    ax = mean_df.plot(subplots=True)
    plt.show()
