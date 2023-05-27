import argparse
import os
import numpy as np
import wandb
from pathlib import Path
import yaml
from itertools import cycle, product, starmap
from functools import partial
import pandas as pd
from typing import Callable, List, Dict, Union
from torch.multiprocessing import get_context

from ai4mat.data.data import (
    StorageResolver,
    DataLoader,
    get_prediction_path,
    Is_Intensive,
    TEST_FOLD,
)

from ai4mat.models import get_predictor_by_name

IS_INTENSIVE = Is_Intensive()


def main():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--trials", type=str, nargs="+")
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument("--gpus", type=int, nargs="+")
    hardware.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--processes-per-unit", type=int, default=1,
                        help="Number of training processes to use per GPU or CPU")
    parser.add_argument("--targets", type=str, nargs="+",
                        help="Only run on these targets")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of jobs for a training process to run. Not all models support this parameter.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save checkpoints for the trained models.")
    parser.add_argument("--output-folder", type=Path, help="Path where to write the output. "
                        "The usual directory structure "
                        "'datasets/predictions/<experiments>/<target>/<trial>.csv.gz'"
                        "will be created.")

    args = parser.parse_args()

    os.environ["WANDB_START_METHOD"] = "thread"
    if "WANDB_RUN_GROUP" not in os.environ:
        os.environ["WANDB_RUN_GROUP"] = "2D-crystal-" + wandb.util.generate_id()
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    else:
        os.environ["WANDB_ENTITY"] = ""
    if args.cpu:
        gpus = [None]
    else:
        gpus = args.gpus
    for experiment_name in args.experiments:
        run_experiment(experiment_name,
                       args.trials,
                       gpus,
                       args.processes_per_unit,
                       args.targets,
                       args.n_jobs,
                       output_folder=args.output_folder,
                       save_checkpoints=args.save_checkpoints)


def run_experiment(experiment_name: str,
                   trials_names: List[str],
                   gpus: List[int],
                   processes_per_unit: int,
                   requested_targets: List[str] = None,
                   n_jobs=1,
                   output_folder: Path = None,
                   save_checkpoints: bool = False
                   ) -> None:
    """
    Runs an experiment.
    Args:
        experiment_name: Name of the experiment.
        trials_names: Names of the trials.
        gpus: List of GPUs to use.
        processes_per_unit: Number of processes to use per GPU.
        requested_targets: List of targets to run on. If None, run on all targets in the experiment.
    Used files and fields:
        experiment - config file, path to the dataset, cv strategy, n folds and targets
        trial - config with model name, representation and model params
    """

    storage_resolver = StorageResolver()
    experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
    with open(Path(experiment_path, "config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    folds = pd.read_csv(Path(experiment_path, "folds.csv.gz"), index_col="_id")
    weights = folds.loc[:, 'weight']\
        if 'weight' in folds.columns else pd.Series(data=np.ones(len(folds.index)), index=folds.index)
    folds = folds.loc[:, 'fold']

    loader = DataLoader(experiment["datasets"], folds.index)

    if requested_targets is None:
        used_targets = experiment["targets"]
    else:
        used_targets = set(experiment["targets"]).intersection(requested_targets)

    for target_name, this_trial_name in product(used_targets, trials_names):
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
        
        minority_class_upsampling = this_trial.get("minority_class_upsampling", False)
        wandb_config = {
            "trial": this_trial,
            "experiment": experiment,
            "target": target_name,
        }
        predictions = cross_val_predict(
            structures,
            targets,
            folds,
            weights,
            get_predictor_by_name(this_trial["model"]),
            IS_INTENSIVE[target_name],
            this_trial["model_params"],
            gpus,
            processes_per_unit,
            wandb_config,
            checkpoint_path=StorageResolver(root_folder=output_folder)["checkpoints"].joinpath(
                experiment_name, str(target_name), this_trial_name),
            n_jobs=n_jobs,
            strategy=experiment['strategy'],
            minority_class_upsampling=minority_class_upsampling,
            save_checkpoints=save_checkpoints
        )
        predictions.rename(lambda target_name: f"predicted_{target_name}_test", axis=1, inplace=True)
        save_path = StorageResolver(root_folder=output_folder)["predictions"].joinpath(
            get_prediction_path(experiment_name, str(target_name), this_trial_name)
        )
        save_path.parents[0].mkdir(exist_ok=True, parents=True)
        predictions.to_csv(save_path, index_label="_id")
        print("Predictions have been saved!", save_path)


def cross_val_predict(
        data: pd.Series,
        targets: Union[pd.Series, List[pd.Series]],
        folds: pd.Series,
        weights: pd.Series,
        predict_func: Callable,
        # predict_func(train, train_targets, test, test_targets, model_params, gpu)
        # returns predictions on test
        # test_targets are used for monitoring
        target_is_intensive: bool,
        model_params: Dict,
        gpus: List[int],
        processes_per_unit: int,
        wandb_config,
        checkpoint_path,
        n_jobs,
        strategy="cv",
        minority_class_upsampling=False,
        save_checkpoints=False
):
    assert data.index.equals(targets.index)
    assert data.index.equals(folds.index)
  
    n_folds = folds.max() + 1
    if strategy == "cv":
        test_fold_generator = range(n_folds)
    elif strategy == "train_test":
        test_fold_generator = (TEST_FOLD,)
    else:
        raise ValueError('Unknown split strategy')
    assert set(folds.unique()) == set(range(n_folds))
    if strategy == "cv":
        # Not necessary, but makes debugging easier
        n_processes = len(gpus) * processes_per_unit
        if n_processes > 1:
            with get_context('spawn').Pool(n_processes, maxtasksperchild=1) as pool:
                predictions = pool.starmap(
                    partial(predict_on_fold,
                            n_folds=n_folds,
                            folds=folds,
                            weights=weights,
                            data=data,
                            targets=targets,
                            predict_func=predict_func,
                            target_is_intensive=target_is_intensive,
                            model_params=model_params,
                            wandb_config=wandb_config,
                            checkpoint_path=checkpoint_path,
                            n_jobs=n_jobs,
                            minority_class_upsampling=minority_class_upsampling,
                            save_checkpoints=save_checkpoints
                            ),
                    zip(test_fold_generator, cycle(gpus)),
                    chunksize=1,
                )
        else:
            predictions = starmap(
                partial(predict_on_fold,
                        n_folds=n_folds,
                        folds=folds,
                        weights=weights,
                        data=data,
                        targets=targets,
                        predict_func=predict_func,
                        target_is_intensive=target_is_intensive,
                        model_params=model_params,
                        wandb_config=wandb_config,
                        checkpoint_path=checkpoint_path,
                        n_jobs=n_jobs,
                        minority_class_upsampling=minority_class_upsampling,
                        save_checkpoints=save_checkpoints
                        ),
                zip(test_fold_generator, cycle(gpus)),
            )
    elif strategy == "train_test":
        predictions = predict_on_fold(
            test_fold=TEST_FOLD,
            gpu=gpus[0],
            n_folds=n_folds,
            folds=folds,
            weights=weights,
            data=data,
            targets=targets,
            predict_func=predict_func,
            target_is_intensive=target_is_intensive,
            model_params=model_params,
            wandb_config=wandb_config,
            checkpoint_path=checkpoint_path,
            n_jobs=n_jobs,
            minority_class_upsampling=minority_class_upsampling,
            save_checkpoints=save_checkpoints
        )
    # TODO(kazeevn)
    # Should we add explicit Structure -> graph preprocessing with results shared?

    if isinstance(targets, pd.DataFrame):
        predictions_pd = pd.DataFrame(index=targets.index, columns=targets.columns,
                                      data=np.zeros_like(targets.to_numpy()))
    elif isinstance(targets, pd.Series):
        predictions_pd = pd.DataFrame(index=targets.index, columns=[targets.name],
                                      data=np.zeros_like(targets.to_numpy()))

    if strategy == "cv":
        if isinstance(targets, pd.DataFrame):
            predictions_pd = pd.DataFrame(index=targets.index, columns=targets.columns,
                                          data=np.zeros_like(targets.to_numpy()))
        elif isinstance(targets, pd.Series):
            predictions_pd = pd.DataFrame(index=targets.index, columns=[targets.name],
                                          data=np.zeros_like(targets.to_numpy()))

        for this_predictions, test_fold in zip(predictions, range(n_folds)):
            test_mask = folds == test_fold
            predictions_pd[test_mask] = this_predictions
    elif strategy == "train_test":
        if len(predictions) > 0:
            predictions_pd = pd.DataFrame(index=folds[folds == TEST_FOLD].index,
                                          columns=[targets.name], data=predictions)
        else:
            predictions_pd = pd.DataFrame(columns=[targets.name])

    return predictions_pd


def predict_on_fold(
        test_fold,
        gpu,
        n_folds,
        folds,
        weights,
        data,
        targets,
        predict_func,
        target_is_intensive,
        model_params,
        wandb_config,
        checkpoint_path,
        n_jobs,
        minority_class_upsampling,
        save_checkpoints
):
    train_folds = set(range(n_folds)) - set((test_fold,))
    train_ids = folds[folds.isin(train_folds)]
    train = data.reindex(index=train_ids.index)
    train_weights = weights.reindex(index=train_ids.index)
    test_ids = folds[folds == test_fold]
    test = data.reindex(index=test_ids.index)
    test_weights = weights.reindex(index=test_ids.index)
    this_wandb_config = wandb_config.copy()
    this_wandb_config["test_fold"] = test_fold
    with wandb.init(
        project="ai4material_design",
        entity=os.environ["WANDB_ENTITY"],
        config=this_wandb_config,
        group=f'{targets.name}'):
        return predict_func(
            train,
            targets.reindex(index=train_ids.index),
            train_weights,
            test,
            targets.reindex(index=test_ids.index),
            test_weights,
            target_is_intensive,
            model_params,
            gpu,
            checkpoint_path=checkpoint_path.joinpath('_'.join(map(str, train_folds))),
            n_jobs=n_jobs,
            minority_class_upsampling=minority_class_upsampling,
            save_checkpoints=save_checkpoints
        )


if __name__ == "__main__":
    main()
