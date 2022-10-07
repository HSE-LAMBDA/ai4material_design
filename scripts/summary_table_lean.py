import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from prettytable import PrettyTable as pt
import re
import sys
sys.path.append('.')
from ai4mat.data.data import StorageResolver, get_prediction_path


def read_results(folds_experiment_name: str,
                 predictions_experiment_name: str,
                 trial:str,
                 skip_missing:bool) -> dict[str, tuple[float]]:
    storage_resolver = StorageResolver()
    folds = pd.read_csv(storage_resolver["experiments"].joinpath(
                        folds_experiment_name).joinpath("folds.csv.gz"),
                        index_col="_id").squeeze('columns')

    experiment_path = storage_resolver["experiments"].joinpath(predictions_experiment_name)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    
    results = {}
    true_targets = pd.concat([pd.read_csv(storage_resolver["processed"]/path/"targets.csv.gz",
                                        index_col="_id",
                                        usecols=["_id"] + experiment["targets"])
                                        for path in experiment["datasets"]], axis=0).reindex(
                                        index=folds.index)
    for target_name in experiment["targets"]:
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
        mae = errors.mean()
        error_std = errors.std()
        mae_cv_std = np.std(errors.groupby(by=folds).mean())
        results[target_name] = (mae, mae_cv_std, error_std)
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
    parser.add_argument("--separate-by", choices=["experiment", "target", "trial"],
        help="Tables are 2D, but we have 3 dimensions: target, trial, experiment. "
        "One of them must be used to separate the tables.")
    parser.add_argument("--presentation-config", type=str)
    parser.add_argument("--skip-missing-data", action="store_true",
                        help="Skip experiments that don't have data for all targets")
    parser.add_argument("--save-pandas", type=Path,
                        help="Save the pandas dataframe to a file")
    args = parser.parse_args()
    
    results = []
    for experiment in args.experiments:
        for trial in args.trials:
            these_results = pd.DataFrame.from_dict(read_results(experiment, experiment, trial, skip_missing=args.skip_missing_data),
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
                these_results = pd.DataFrame.from_dict(read_results(experiment, args.combined_experiment, trial, skip_missing=args.skip_missing_data),
                                                       orient="index", columns=["MAE", "MAE_CV_std", "error_std"])
                these_results['experiment'] = experiment + "_combined"
                these_results['trial'] = trial
                these_results.index.name = "target"
                these_results.set_index(["experiment", "trial"], inplace=True, append=True)
                results.append(these_results)
    results_pd = pd.concat(results, axis=0)
    if args.save_pandas:
        results_pd.to_pickle(args.save_pandas)
    
    results_str = results_pd.apply(lambda x: f"{x['MAE']:.3f} ± {x['MAE_CV_std']:.3f}", axis="columns")

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
        columns = "trial"
    else:
        raise ValueError("Must separate by one of experiment, trial, target")
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