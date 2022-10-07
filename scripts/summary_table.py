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


def main():
    parser = argparse.ArgumentParser("Makes a text table with MAEs")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--combined-experiment", type=str)
    parser.add_argument("--trials", type=str, nargs="+")
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
    args = parser.parse_args()

    results = []
    for experiment in args.experiments:
        for trial in args.trials:
            these_results = pd.DataFrame.from_dict(read_results(experiment, experiment, trial),
                                                   orient="index", columns=["MAE", "MAE_CV_std", "error_std"])
            these_results['experiment'] = experiment
            these_results['trial'] = trial
            these_results.index.name = "target"
            these_results.set_index(["experiment", "trial"], inplace=True, append=True)
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
    print(results_pd)

    if args.presentation_config:
        with open(args.presentation_config) as config_file:
            presentatation_config = yaml.safe_load(config_file)
    else:
        presentatation_config = None

    all_per_spin_targets = ("homo", "lumo", "band_gap")
    if args.separate_by == "trial":
        for trial in args.trials:
            table_data = results_pd.loc[:, :, trial]
            present_targets = table_data.index.get_level_values("target").unique()
            per_spin_targets = present_targets.intersection(all_per_spin_targets)
            # Add None for missing values
            new_index = pd.MultiIndex.from_product(table_data.index.remove_unused_levels().levels)
            table_data = table_data.reindex(new_index)
            if args.populate_per_spin_target:
                for spin in ("majority", "minority"):
                    for target in per_spin_targets:
                        if target not in present_targets:
                            continue
                        table_data.loc[f"{target}_{spin}"].update(
                            table_data.xs(f"{target}_{spin}").fillna(table_data.xs(target)))
                table_data.drop(list(per_spin_targets), level="target", inplace=True)
                table_data.index = table_data.index.remove_unused_levels()
                present_targets = table_data.index.get_level_values("target").unique()
            if args.targets:
                table_data = table_data.loc[args.targets]
                present_targets = table_data.index.get_level_values("target").unique()
            mae_table = pt()
            mae_table.field_names = [args.experiment_column] + list(present_targets)
            experiment_order = table_data.index.get_level_values("experiment").unique()
            if args.combined_experiment and args.combined_experiment in experiment_order:
                experiment_order = [args.combined_experiment] + list(experiment_order.drop(args.combined_experiment))
            for experiment in experiment_order:
                table_row = [experiment.split("/")[-1].replace("_500", "")]
                for _, row in table_data.loc[(slice(None), experiment), :].iterrows():
                    target = row.name[0]
                    if (presentatation_config is not None and
                            target in presentatation_config and
                            "presentation_multiplier" in presentatation_config[target]):
                        unit_multiplier = presentatation_config[target]["presentation_multiplier"]
                    else:
                        unit_multiplier = 1
                    table_row.append(f"{row['MAE'] * unit_multiplier:.3f} ± "
                                     f"{row['MAE_CV_std'] * unit_multiplier:.3f}")
                mae_table.add_row(table_row)
        print(mae_table)


if __name__ == "__main__":
    main()
