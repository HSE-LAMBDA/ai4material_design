import argparse
import os

import yaml
import pandas as pd
import numpy as np
import logging
import sys

sys.path.append('.')
from ai4mat.data.data import StorageResolver, get_prediction_path


def read_results(folds_experiment_name: str,
                 predictions_experiment_name: str,
                 trial: str):
    storage_resolver = StorageResolver()
    folds = pd.read_csv(storage_resolver["experiments"].joinpath(
        folds_experiment_name).joinpath("folds.csv.gz"),
                        index_col="_id")
    weights = folds.loc[:, 'weight'] if 'weight' in folds.columns else pd.Series(data=np.ones((len(folds))), index=folds.index)
    folds = folds.loc[:, 'fold']

    experiment_path = storage_resolver["experiments"].joinpath(predictions_experiment_name)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)

    results = []
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
        mae = (errors * weights).sum() / weights.sum()
        results.append(mae)

    return results, experiment['targets']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--trials-folder", type=str, required=True)
    parser.add_argument("--skip-missing-data", action="store_true",
                        help="Skip experiments that don't have data for all targets")
    args = parser.parse_args()

    sr = StorageResolver()
    trial_folder = sr['trials'].joinpath(args.trials_folder)
    experiment = args.experiment

    results = {}
    for trial in os.listdir(trial_folder):
        try:
            these_results, targets = read_results(experiment, experiment, args.trials_folder + '/' + trial[:-5])

        except FileNotFoundError:
            if args.skip_missing_data:
                logging.warning("Skipping expriment %s; trial %s because it doesn't have data for all targets",
                                experiment, trial)
                continue
            else:
                raise

        results[trial[:-5]] = these_results

    df_results = pd.DataFrame.from_dict(results, orient='index', columns=[targets])
    print(df_results)

    folds = pd.read_csv(sr["experiments"].joinpath(experiment).joinpath("folds.csv.gz"), index_col="_id").loc[:, 'fold']
    experiment_path = sr["experiments"].joinpath(experiment)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    for target in targets:
        cur_col = df_results.loc[:, target].squeeze()
        best_trial = cur_col.index[cur_col.argmin()]
        print(f"for {target} best trial is {best_trial} with mae {cur_col[cur_col.argmin()]}")
        print(f"{target} mae data")

        for dataset in experiment['datasets']:
            cur_targets = pd.read_csv(sr["processed"] / dataset / "targets.csv.gz", index_col="_id").reindex(folds.index)
            predictions = pd.read_csv(sr["predictions"].joinpath(
                get_prediction_path(
                    experiment,
                    target,
                    best_trial
                )), index_col="_id").squeeze("columns")

            mae = np.abs(predictions - cur_targets.loc[:, target])
            print(f"{dataset} {mae}")


if __name__ == "__main__":
    main()
