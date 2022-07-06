from itertools import product
from collections import defaultdict
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from prettytable import PrettyTable as pt
import sys
sys.path.append('.')
from ai4mat.data.data import StorageResolver, get_prediction_path, get_targets_path


def print_target_trial_table(experiment, trials, unit_multiplier):
    storage_resolver = StorageResolver()
    experiment_path = storage_resolver["experiments"].joinpath(experiment)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    folds = pd.read_csv(experiment_path.joinpath("folds.csv"),
                        index_col="_id",
                        squeeze=True)
    # Support running on a part of the dataset, defined via folds
    true_targets = pd.concat([pd.read_csv(get_targets_path(path), index_col="_id")
                                for path in experiment["datasets"]], axis=0).reindex(
                                        index=folds.index)
    mae_table = pt()
    mae_table.field_names = ["Model"] + experiment["targets"]
    for this_trial_name in trials:
        row = [this_trial_name]
        for target_name in experiment["targets"]:
            predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
                                            get_prediction_path(
                                                experiment,
                                                target_name,
                                                this_trial_name
                                            )), index_col="_id", squeeze=True)
            assert predictions.index.equals(true_targets.index)
            errors = np.abs(predictions - true_targets.loc[:, target_name])
            mae = errors.mean()
            mae_cv_std = np.std(errors.groupby(by=folds).mean())
            row.append(f"{mae*unit_multiplier:.1f} ± {mae_cv_std*unit_multiplier:.1f}")
        mae_table.add_row(row)
    print(mae_table)


def print_experiment_trial_table(experiments, trials, target_name, unit_multiplier):
    storage_resolver = StorageResolver()
    mae_table = pt()
    mae_table.field_names = ["Experiment"] + trials
    for experiment_name in experiments:
        row = [experiment_name]
        experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
        with open(experiment_path.joinpath("config.yaml")) as experiment_file:
            experiment = yaml.safe_load(experiment_file)
        folds = pd.read_csv(experiment_path.joinpath("folds.csv"),
                            index_col="_id",
                            squeeze=True)
        true_targets = pd.concat([pd.read_csv(get_targets_path(path), index_col="_id",
                                    usecols=["_id", target_name]).squeeze("columns")
                                    for path in experiment["datasets"]], axis=0).reindex(
                                    index=folds.index)
        for trial in trials:
            predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
                                            get_prediction_path(
                                                experiment_name,
                                                target_name,
                                                trial
                                            )), index_col="_id", squeeze=True)
            these_targets = true_targets.reindex(index=predictions.index)
            errors = np.abs(predictions - these_targets)
            mae = errors.mean()
            row.append(f"{mae*unit_multiplier:.1f}")
        mae_table.add_row(row)
    print(mae_table)


def main():
    parser = argparse.ArgumentParser("Makes a text table with MAEs")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--trials", type=str, nargs="+")
    parser.add_argument("--separate-by", choices=["experiment", "target"],
        help="Tables are 2D, but we have 3 dimensions: target, trial, experiment. "
        "One of them must be used to separate the tables.")
    parser.add_argument("--unit-multiplier", type=float, default=1e3,
        help="As of May 2022, data in the project are in eV, so the 1000 mutliplier makes it meV")
    args = parser.parse_args()
    
    if args.separate_by == "experiment":
        for experiment in args.experiments:
            print_target_trial_table(experiment, args.trials, args.unit_multiplier)
    elif args.separate_by == "target":
        storage_resolver = StorageResolver()
        experiment_path = storage_resolver["experiments"].joinpath(args.experiments[0])
        with open(experiment_path.joinpath("config.yaml")) as experiment_file:
            experiment = yaml.safe_load(experiment_file)
        targets = experiment["targets"]
        for target_name in targets:
            print(f"{target_name}:")
            print_experiment_trial_table(
                args.experiments, args.trials, target_name, args.unit_multiplier)
   

if __name__ == "__main__":
    main()
