from itertools import product
from collections import defaultdict
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging
from prettytable import PrettyTable as pt
import sys

sys.path.append('.')
from ai4mat.data.data import StorageResolver, get_prediction_path, get_targets_path


def name_to_train_WSe2_count(name):
    fraction = float(name.split("_")[-1])
    return int(5934 * (1. - fraction))


def read_results(folds_experiment_name: str,
                 predictions_experiment_name: str,
                 trial: str):
    storage_resolver = StorageResolver()
    folds = pd.read_csv(storage_resolver["experiments"].joinpath(
        folds_experiment_name).joinpath("folds.csv.gz"),
                        index_col="_id").squeeze('columns')

    experiment_path = storage_resolver["experiments"].joinpath(predictions_experiment_name)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)

    results = {}
    true_targets = pd.concat([pd.read_csv(storage_resolver["processed"] / path / "targets.csv.gz",
                                          index_col="_id",
                                          usecols=["_id"] + experiment["targets"])
                              for path in experiment["datasets"]], axis=0).reindex(
        index=folds.index)
    for target_name in experiment["targets"]:
        predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
            get_prediction_path(
                predictions_experiment_name,
                target_name,
                trial
            )), index_col="_id").squeeze("columns")
        errors = np.abs(predictions - true_targets.loc[:, target_name])
        mae = errors.mean()
        error_std = errors.std()
        mae_cv_std = np.std(errors.groupby(by=folds).mean())
        results[target_name] = (mae, mae_cv_std, error_std)
    return results


def find_in_dict(key, src):
    for k, value in src.items():
        if key == k:
            return value
        elif type(value) == dict:
            res = find_in_dict(key, value)
            if res is not None:
                return res
    return



def main():
    parser = argparse.ArgumentParser("Makes a text table with MAEs")
    parser.add_argument("--experiments", type=str, nargs="+", required=True)
    parser.add_argument("--combined-experiment", type=str)
    parser.add_argument("--trials", type=str, nargs="+", required=True)
    parser.add_argument("--targets", type=str, nargs="+")
    parser.add_argument("--separate-by", choices=["experiment", "target", "trial"],
                        help="Tables are 2D, but we have 3 dimensions: target, trial, experiment. "
                             "One of them must be used to separate the tables.")
    parser.add_argument("--presentation-config", type=str)
    parser.add_argument("--experiment-name-converter", type=str,
                        help="Name of a local function to convert experiment names to for human-readable display")
    parser.add_argument("--experiment_column", type=str, default="Experiment",
                        help="Name of the column in the table that corresponds to experiment")
    parser.add_argument("--populate-per-spin-target", action="store_true",
                        help="Populate {band_gap,homo,lumo}_{majority,minority} columns with"
                             " values from the non-spin-specific versions")
    parser.add_argument("--skip-missing-data", action="store_true",
                        help="Skip experiments that don't have data for all targets")
    parser.add_argument("--trial-parameters-names", type=str, nargs="+")
    args = parser.parse_args()

    sr = StorageResolver()
    trials_short_names = {}
    for trial in args.trials:
        trial_path = trial + '.yaml'
        with open(sr['trials'] / trial_path, 'r') as trial_file:
            cur_trial = yaml.safe_load(trial_file)
            cur_name = ""
            for param_name in args.trial_parameters_names:
                param_value = find_in_dict(param_name, cur_trial)
                cur_name += str(param_value) + ' '
        trials_short_names[trial] = cur_name

    results = []
    for experiment in args.experiments:
        for trial in args.trials:
            try:
                these_results = pd.DataFrame.from_dict(read_results(experiment, experiment, trial),
                                                       orient="index", columns=["MAE", "MAE_CV_std", "error_std"])
            except FileNotFoundError:
                if args.skip_missing_data:
                    logging.warning("Skipping expriment %s; trial %s because it doesn't have data for all targets",
                                    experiment, trial)
                    continue
                else:
                    raise
            these_results['experiment'] = experiment
            these_results['trial'] = trial

            these_results['trial_short_name'] = trials_short_names[trial]

            these_results.index.name = "target"
            these_results.set_index(["experiment", "trial_short_name"], inplace=True, append=True)
            results.append(these_results)

    if args.combined_experiment:
        for experiment in args.experiments:
            if experiment == args.combined_experiment:
                continue
            for trial in args.trials:
                these_results = pd.DataFrame.from_dict(read_results(experiment, args.combined_experiment, trial),
                                                       orient="index", columns=["MAE", "MAE_CV_std", "error_std"])
                these_results['experiment'] = experiment + "_combined"
                these_results['trial'] = trial
                these_results.index.name = "target"
                these_results.set_index(["experiment", "trial"], inplace=True, append=True)
                results.append(these_results)
    results_pd = pd.concat(results, axis=0)

    if args.presentation_config:
        with open(args.presentation_config) as config_file:
            presentatation_config = yaml.safe_load(config_file)
    else:
        presentatation_config = None

    if args.separate_by == "trial":
        rows = "experiment"
        columns = "target"
    elif args.separate_by == "experiment":
        rows = "trial"
        columns = "target"
    elif args.separate_by == "target":
        rows = "experiment"
        columns = "trial_short_name"
    else:
        raise ValueError("Must separate by one of experiment, trial, target")
    all_separators = results_pd.index.get_level_values(args.separate_by).unique()

    for table_index in all_separators:
        table_data = results_pd.xs(table_index, level=args.separate_by)
        # Add None for missing values
        new_index = pd.MultiIndex.from_product(table_data.index.remove_unused_levels().levels)
        table_data = table_data.reindex(new_index)
        mae_table = pt()
        mae_table.field_names = [rows] + list(table_data.index.get_level_values(columns).unique())
        for row_name in table_data.index.get_level_values(rows).unique():
            table_row = [row_name]
            for column_name, cell_value in table_data.xs(row_name, level=rows).iterrows():
                table_row.append(f"{cell_value['MAE']:.3f}") #± "
                                 #f"{cell_value['MAE_CV_std']:.3f}")
            mae_table.add_row(table_row)
        print(table_index)
        print(mae_table)


if __name__ == "__main__":
    main()
