from typing import Dict, Tuple, List
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from prettytable import PrettyTable as pt
import re
import sys
from collections import defaultdict
sys.path.append('.')
from ai4mat.data.data import StorageResolver, get_prediction_path, TEST_FOLD


def read_results(folds_experiment_name: str,
                 predictions_experiment_name: str,
                 trial:str,
                 skip_missing:bool,
                 targets: List[str]) -> Dict[str, Dict[str, float]]:
    storage_resolver = StorageResolver()
    with open(storage_resolver["experiments"].joinpath(folds_experiment_name).joinpath("config.yaml")) as experiment_file:
        folds_yaml = yaml.safe_load(experiment_file)
    folds_definition = pd.read_csv(storage_resolver["experiments"].joinpath(
                        folds_experiment_name).joinpath("folds.csv.gz"),
                        index_col="_id")
    if folds_yaml['strategy'] == 'train_test':
        folds_definition = folds_definition[folds_definition['fold'] == TEST_FOLD]

    folds = folds_definition.loc[:, 'fold']
    weights = folds_definition.loc[:, 'weight']
    
    experiment_path = storage_resolver["experiments"].joinpath(predictions_experiment_name)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)

    results = defaultdict(dict)
    targets_per_dataset = [pd.read_csv(storage_resolver["processed"]/path/"targets.csv.gz",
                                       index_col="_id",
                                       usecols=["_id"] + experiment["targets"])
                                       for path in experiment["datasets"]]
    true_targets = pd.concat(targets_per_dataset, axis=0).reindex(index=folds.index)

    for target_name in set(experiment["targets"]).intersection(targets):
        try:
            predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
                                      get_prediction_path(
                                      predictions_experiment_name,
                                      target_name,
                                      trial
                                      )), index_col="_id").squeeze("columns")
        except FileNotFoundError:
            if skip_missing:
                logging.warning("No predictions for experiment %s; trial %s; target %s",
                                predictions_experiment_name, trial, target_name)
                continue
            else:
                raise
        errors = np.abs(predictions - true_targets.loc[:, target_name])
        mae = np.average(errors, weights=weights)
        results[target_name]['combined'] = mae
        for dataset, targets in zip(experiment["datasets"], targets_per_dataset):
            this_errors = errors.reindex(index=targets.index.intersection(errors.index))
            # Assume the weight is the same for all structures in a dataset
            results[target_name][dataset] = this_errors.mean()
    return results


def main():
    parser = argparse.ArgumentParser("Makes a text table with MAEs")
    parser.add_argument("--experiments", type=str, nargs="+", required=True)
    parser.add_argument("--trials", type=str, nargs="+", required=True)
    parser.add_argument("--combined-experiment", type=str)
    parser.add_argument("--targets", type=str, nargs="+")
    parser.add_argument("--column-format-re", type=re.compile,
                        help="Regular expression to be matched against the column names for formating.")
    parser.add_argument("--row-format-re", type=re.compile,
                        help="Regular expression to be matched against the row names for formating.")
    parser.add_argument("--separate-by", type=str,
        help="Tables are 2D, we must slice the data")
    parser.add_argument("--presentation-config", type=str)
    parser.add_argument("--skip-missing-data", action="store_true",
                        help="Skip experiments that don't have data for all targets")
    parser.add_argument("--save-pandas", type=Path,
                        help="Save the pandas dataframe to a file")
    args = parser.parse_args()
    
    results = []
    for experiment in args.experiments:
        for trial in args.trials:
            these_results = read_results(experiment, experiment, trial, skip_missing=args.skip_missing_data, targets=args.targets)
            these_results_unwrapped = []
            for target, target_results in these_results.items():
                for dataset, mae in target_results.items():
                    these_results_pd = these_results_unwrapped.append({
                        "trial": trial,
                        "target": target,
                        "dataset": dataset,
                        "mae": mae
                    })
            these_results_pd = pd.DataFrame.from_records(these_results_unwrapped)
            these_results_pd.set_index(["target", "dataset", "trial"], inplace=True)
            results.append(these_results_pd)
    
    results_pd = pd.concat(results, axis=0)
    if args.save_pandas:
        results_pd.to_pickle(args.save_pandas)
    
    results_str = results_pd["mae"].apply(lambda x: f"{x:.3f}")

    if args.presentation_config:
        with open(args.presentation_config) as config_file:
            presentatation_config = yaml.safe_load(config_file)
    else:
        presentatation_config = None
    
    table_keys = [name for name in results_str.index.names if name not in args.separate_by]
    rows, columns = table_keys
    all_separators = results_str.index.get_level_values(args.separate_by).unique()

    for table_index in all_separators:
        table_data = results_str.xs(table_index, level=args.separate_by)
        # Add None for missing values
        new_index = pd.MultiIndex.from_product(table_data.index.remove_unused_levels().levels)
        table_data = table_data.reindex(new_index)
        mae_table = pt()
        column_names = list(table_data.index.get_level_values(columns).unique())
        if args.column_format_re:
            column_names = [args.column_format_re.match(name).group("name") for name in column_names]
        mae_table.field_names = [rows] + column_names
        for row_name in sorted(table_data.index.get_level_values(rows).unique()):
            if args.row_format_re:
                table_row = [args.row_format_re.match(row_name).group("name")]
            else:
                table_row = [row_name]
            for column_name, cell_value in table_data.xs(row_name, level=rows).items():
                table_row.append(cell_value)
            mae_table.add_row(table_row)
        print(table_index)
        print(mae_table)


if __name__ == "__main__":
    main()