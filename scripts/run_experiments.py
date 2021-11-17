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
from multiprocessing import Pool

from data import get_pickle_path, get_column_from_data_type, IS_INTENSIVE
from models import get_predictor_by_name


def main():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--trials", type=str, nargs="+")
    parser.add_argument("--predictions-root", type=str, required=True)
    parser.add_argument("--gpus", type=int, nargs="*")
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    args = parser.parse_args()
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_RUN_GROUP"] = "2D-crystal-" + wandb.util.generate_id()
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    for experiment_path in args.experiments:
        run_experiment(experiment_path,
                       args.trials,
                       args.predictions_root,
                       args.gpus,
                       args.processes_per_gpu)


def run_experiment(experiment_path, trials_paths, output_path, gpus, processes_per_gpu):
    with open(Path(experiment_path, "config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)

    data = pd.concat([pd.read_pickle(get_pickle_path(path))
                      for path in experiment["datasets"]], axis=0)
    folds = pd.read_csv(Path(experiment_path, "folds.csv"),
                        index_col="_id",
                        squeeze=True)
    if "ignore-missing" in experiment and experiment["ignore-missing"]:
        folds = folds.reindex(index=data.index)
    
    for target_name, this_trial_path in product(experiment["targets"], trials_paths):
        with open(this_trial_path) as this_trial_file:
            this_trial = yaml.safe_load(this_trial_file)
        structures = data[get_column_from_data_type(this_trial["representation"])]
        wandb_config = {"trial": this_trial,
                        "experiment": experiment}

        predictions = cross_val_predict(structures,
                                        data.loc[:, target_name],
                                        folds,
                                        get_predictor_by_name(this_trial["model"]),
                                        IS_INTENSIVE[target_name],
                                        this_trial["model_params"],
                                        gpus,
                                        processes_per_gpu,
                                        wandb_config)
        predictions.rename(f"predicted_f{target}_test")
        save_path = Path(output_path,
                         get_experiment_name(experiment_path),
                         target_name)
        save_path.mkdir(exist_ok=True, parents=True)
        predictions.to_csv(Path(save_path, f"{get_trial_name(this_trial_path)}.csv"),
                           index_label="_id")


def cross_val_predict(data: pd.Series,
                      targets: pd.Series,
                      folds: pd.Series,
                      predict_func: Callable,
                      # predict_func(train, train_targets, test, test_targets, model_params, gpu)
                      # returns (train_predictions, test_predictions)
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

    with Pool(len(gpus)*processes_per_gpu) as pool:
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
    # Should we add explicit Structure->graph preprocessing with results shared?
    predictions_pd = pd.Series(index=targets.index, data=np.empty_like(targets.array))

    for this_predictions, test_fold in zip(predictions, range(n_folds)):
        test_mask = folds[folds == test_fold]
        predictions_pd[test_mask] = this_predictions[1][test_mask]

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
                    config=this_wandb_config) as run:
        return predict_func(train, targets.reindex(index=train_ids.index),
                            test, targets.reindex(index=test_ids.index),
                            target_is_intensive,
                            model_params, gpu)


if __name__ == "__main__":
    main()
