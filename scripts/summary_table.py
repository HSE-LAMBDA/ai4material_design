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
    return int(5934*(1. - fraction))


def read_results(experiment_name, trial):
    storage_resolver = StorageResolver()
    experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    folds = pd.read_csv(experiment_path.joinpath("folds.csv.gz"),
                        index_col="_id").squeeze('columns')
    results = {}
    true_targets = pd.concat([pd.read_csv(storage_resolver["processed"]/path/"targets.csv.gz",
                                        index_col="_id",
                                        usecols=["_id"] + experiment["targets"])
                                        for path in experiment["datasets"]], axis=0).reindex(
                                        index=folds.index)
    for target_name in experiment["targets"]:
        predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
                                    get_prediction_path(
                                        experiment_name,
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
    parser.add_argument("--trials", type=str, nargs="+")
    parser.add_argument("--targets", type=str, nargs="+")
    parser.add_argument("--separate-by", choices=["experiment", "target", "trial"],
        help="Tables are 2D, but we have 3 dimensions: target, trial, experiment. "
        "One of them must be used to separate the tables.")
    parser.add_argument("--unit-multiplier", type=float, default=1e3,
        help="As of May 2022, data in the project are in eV, so the 1000 mutliplier makes it meV")
    parser.add_argument("--experiment-name-converter", type=str,
        help="Name of a local function to convert experiment names to for human-readable display")
    parser.add_argument("--experiment_column", type=str, default="Experiment",
                       help="Name of the column in the table that corresponds to experiment")
    parser.add_argument("--parameter-to-extract", type=str, help="path to parameter to extract for table in format "
                                                                 "a/b/c")
    parser.add_argument("--populate-per-spin-target", action="store_true",
                        help="Populate {band_gap,homo,lumo}_{majority,minority} columns with"
                        " values from the non-spin-specific versions")
    args = parser.parse_args()
    
    results = []
    for experiment in args.experiments:
        for trial in args.trials:
            these_results = pd.DataFrame.from_dict(read_results(experiment, trial),
                                                   orient="index", columns=["MAE", "MAE_CV_std", "error_std"])
            these_results['experiment'] = experiment
            these_results['trial'] = trial
            these_results.index.name = "target"
            these_results.set_index(["experiment", "trial"], inplace=True, append=True)
            results.append(these_results)
    results_pd = pd.concat(results, axis=0)

    if args.targets:
        results_pd = results_pd.loc[args.targets]
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
            mae_table = pt()
            mae_table.field_names = [args.experiment_column] + list(present_targets)
            for experiment in table_data.index.get_level_values("experiment").unique():
                table_row = [experiment.split("/")[-1][:-4]]
                for _, row in table_data.loc[(slice(None), experiment), :].iterrows():
                    table_row.append(f"{row['MAE']*args.unit_multiplier:.1f} ± "
                    f"{row['MAE_CV_std']*args.unit_multiplier:.1f}")
                mae_table.add_row(table_row)
        print(mae_table)


if __name__ == "__main__":
    main()
