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

from ai4mat.data.data import (
    StorageResolver,
    get_column_from_data_type,
    get_prediction_path,
    IS_INTENSIVE,
    get_experiment_name)

from ai4mat.models import get_predictor_by_name




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
        kwargs['context'] = NoDaemonContext()
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
        run_experiment(experiment_name,
                       args.trials,
                       args.gpus,
                       args.processes_per_gpu)


def run_experiment(experiment_name, trials_names, gpus, processes_per_gpu):
    storage_resolver = StorageResolver()
    experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
    with open(Path(experiment_path, "config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    folds = pd.read_csv(Path(experiment_path, "folds.csv"),
                        index_col="_id",
                        squeeze=True)
    # Support running on a part of the dataset, defined via folds
    data = pd.concat([pd.read_pickle(storage_resolver["processed"].joinpath(
        get_experiment_name(path), "data.pickle.gz"))
                      for path in experiment["datasets"]], axis=0).reindex(index=folds.index)

    for target_name, this_trial_name in product(experiment["targets"], trials_names):
        with open(storage_resolver["trials"].joinpath(f"{this_trial_name}.yaml")) as this_trial_file:
            this_trial = yaml.safe_load(this_trial_file)
        structures = data[get_column_from_data_type(this_trial["representation"])]
        # State stores the material composition and is semi-supported by pymatgen
        # so we ensure it's present
        if this_trial["representation"] == "sparse":
            assert getattr(structures.iloc[0], "state", None) is not None
        wandb_config = {"trial": this_trial,
                        "experiment": experiment,
                        "target": target_name}

        predictions = cross_val_predict(structures,
                                        data.loc[:, target_name],
                                        folds,
                                        get_predictor_by_name(this_trial["model"]),
                                        IS_INTENSIVE[target_name],
                                        this_trial["model_params"],
                                        gpus,
                                        processes_per_gpu,
                                        wandb_config)
        predictions.rename(f"predicted_{target_name}_test", inplace=True)
        save_path = storage_resolver["predictions"].joinpath(
                         get_prediction_path(
                             experiment_name,
                             target_name,
                             this_trial_name
                         ))
        save_path.parents[0].mkdir(exist_ok=True, parents=True)
        predictions.to_csv(save_path, index_label="_id")


def cross_val_predict(data: pd.Series,
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
                      wandb_config):
    assert data.index.equals(targets.index)
    assert data.index.equals(folds.index)

    n_folds = folds.max() + 1
    assert set(folds.unique()) == set(range(n_folds))

    with NestablePool(len(gpus) * processes_per_gpu) as pool:

        predictions = pool.starmap(partial(
            predict_on_fold,
            n_folds=n_folds,
            folds=folds,
            data=data,
            targets=targets,
            predict_func=predict_func,
            target_is_intensive=target_is_intensive,
            model_params=model_params,
            wandb_config=wandb_config
        ), zip(range(n_folds), cycle(gpus)))
    # TODO(kazeevn)
    # Should we add explicit Structure -> graph preprocessing with results shared?
    predictions_pd = pd.Series(index=targets.index, data=np.empty_like(targets.array))

    for this_predictions, test_fold in zip(predictions, range(n_folds)):
        test_mask = (folds == test_fold)
        predictions_pd[test_mask] = this_predictions

    return predictions_pd


def predict_on_fold(test_fold,
                    gpu,
                    n_folds,
                    folds,
                    data,
                    targets,
                    predict_func,
                    target_is_intensive,
                    model_params,
                    wandb_config):
    train_folds = set(range(n_folds)) - set((test_fold,))
    train_ids = folds[folds.isin(train_folds)]
    train = data.reindex(index=train_ids.index)
    test_ids = folds[folds == test_fold]
    test = data.reindex(index=test_ids.index)
    this_wandb_config = wandb_config.copy()
    this_wandb_config["test_fold"] = test_fold
    with wandb.init(project='ai4material_design',
                    entity=os.environ["WANDB_ENTITY"],
                    config=this_wandb_config, 
                    # mode="disabled",
                    ) as run: 
        return predict_func(train, targets.reindex(index=train_ids.index),
                            test, targets.reindex(index=test_ids.index),
                            target_is_intensive,
                            model_params, gpu)


if __name__ == "__main__":
    main()
