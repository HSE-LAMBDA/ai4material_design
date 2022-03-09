import argparse
import os
import numpy as np
import wandb
from pathlib import Path
import yaml
from itertools import cycle, product
from functools import partial
import pandas as pd
from typing import Callable, List, Dict
import multiprocessing.pool
import multiprocessing

from ai4mat.data.data import (
    StorageResolver,
    DataLoader,
    get_prediction_path,
    IS_INTENSIVE,
)

from ai4mat.models import get_predictor_by_name

# This might have unexpected effects, haven't been tested on pytorch yet!
multiprocessing.set_start_method('fork', force=True)
# This should be moved to somewhere else probaby utils
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--trials", type=str, nargs="+")
    parser.add_argument("--gpus", type=int, nargs="*")
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    args = parser.parse_args()

    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_RUN_GROUP"] = "2D-crystal-" + wandb.util.generate_id()
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    for experiment_name in args.experiments:
        run_experiment(experiment_name, args.trials, args.gpus, args.processes_per_gpu)


def run_experiment(experiment_name, trials_names, gpus, processes_per_gpu):
    # used variables:
    # experiment - config file, path to the dataset, cv strategy, n folds and targets
    # this trial - config with model name, representation and model params
    storage_resolver = StorageResolver()
    experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
    with open(Path(experiment_path, "config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    folds = pd.read_csv(
        Path(experiment_path, "folds.csv"), index_col="_id", squeeze=True
    )

    loader = DataLoader(experiment["datasets"], folds.index)

    for target_name, this_trial_name in product(experiment["targets"], trials_names):
        with open(
            storage_resolver["trials"].joinpath(f"{this_trial_name}.yaml")
        ) as this_trial_file:
            this_trial = yaml.safe_load(this_trial_file)

        # loader.get_structures can return both sparse and full features
        # or matminer features (which are not really structures) depending on
        # inquired representation
        structures = loader.get_structures(this_trial["representation"])
        targets = loader.get_targets(target_name)
        # State stores the material composition and is semi-supported by pymatgen
        # so we ensure it's present
        if this_trial["representation"] == "sparse":
            assert getattr(structures.iloc[0], "state", None) is not None

        if this_trial["representation"] == "matminer":
            assert (
                this_trial["model"] == "catboost"
            ), "Expected model 'catboost' for representation 'matminer'"
        if this_trial["model"] == "catboost":
            assert (
                this_trial["representation"] == "matminer"
            ), "Expected representation 'matminer' for model 'catboost'"
        wandb_config = {
            "trial": this_trial,
            "experiment": experiment,
            "target": target_name,
        }

        predictions = cross_val_predict(
            structures,
            targets,
            folds,
            get_predictor_by_name(this_trial["model"]),
            IS_INTENSIVE[target_name],
            this_trial["model_params"],
            gpus,
            processes_per_gpu,
            wandb_config,
            checkpoint_path=storage_resolver["checkpoints"].joinpath(experiment_name, target_name, this_trial_name)
        )
        predictions.rename(f"predicted_{target_name}_test", inplace=True)
        save_path = storage_resolver["predictions"].joinpath(
            get_prediction_path(experiment_name, target_name, this_trial_name)
        )
        save_path.parents[0].mkdir(exist_ok=True, parents=True)
        predictions.to_csv(save_path, index_label="_id")
        print("Predictions has been saved!", save_path)


def cross_val_predict(
    data: pd.Series,
    targets: pd.Series,
    folds: pd.Series,
    predict_func: Callable,
    # predict_func(train, train_targets, test, test_targets, model_params, gpu)
    # returns predictions on test
    # test_targets are used for monitoring
    target_is_intensive: bool,
    model_params: Dict,
    gpus: List[int],
    processes_per_gpu: int,
    wandb_config,
    checkpoint_path,
):
    assert data.index.equals(targets.index)
    assert data.index.equals(folds.index)

    n_folds = folds.max() + 1
    assert set(folds.unique()) == set(range(n_folds))
    with NestablePool(len(gpus) * processes_per_gpu, maxtasksperchild=1) as pool:
        predictions = pool.starmap(
            partial(
                predict_on_fold,
                n_folds=n_folds,
                folds=folds,
                data=data,
                targets=targets,
                predict_func=predict_func,
                target_is_intensive=target_is_intensive,
                model_params=model_params,
                wandb_config=wandb_config,
                checkpoint_path=checkpoint_path,
            ),
            zip(range(n_folds), cycle(gpus)),
        )
    # TODO(kazeevn)
    # Should we add explicit Structure -> graph preprocessing with results shared?
    predictions_pd = pd.Series(index=targets.index, data=np.empty_like(targets.array))

    for this_predictions, test_fold in zip(predictions, range(n_folds)):
        test_mask = folds == test_fold
        predictions_pd[test_mask] = this_predictions

    return predictions_pd


def predict_on_fold(
    test_fold,
    gpu,
    n_folds,
    folds,
    data,
    targets,
    predict_func,
    target_is_intensive,
    model_params,
    wandb_config,
    checkpoint_path,
):
    train_folds = set(range(n_folds)) - set((test_fold,))
    train_ids = folds[folds.isin(train_folds)]
    train = data.reindex(index=train_ids.index)
    test_ids = folds[folds == test_fold]
    test = data.reindex(index=test_ids.index)
    this_wandb_config = wandb_config.copy()
    this_wandb_config["test_fold"] = test_fold

    with wandb.init(
        project="ai4material_design_final_run",
        entity=os.environ["WANDB_ENTITY"],
        config=this_wandb_config,
        # name="full-eos",
        # tags=['full', 'eos']
    ) as run:
        return predict_func(
            train,
            targets.reindex(index=train_ids.index),
            test,
            targets.reindex(index=test_ids.index),
            target_is_intensive,
            model_params,
            gpu,
            checkpoint_path=checkpoint_path.joinpath('_'.join(map(str, train_folds))),
            use_last_checkpoint=True,
        )


if __name__ == "__main__":
    main()
