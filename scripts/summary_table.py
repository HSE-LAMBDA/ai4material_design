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


def main():
    parser = argparse.ArgumentParser("Makes a text experiment/trial table")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--trials", type=str, nargs="+")
    parser.add_argument("--unit-multiplier", type=float, default=1e3,
        help="As of May 2022, data in the project are in eV, so the 1000 mutliplier makes it meV")
    args = parser.parse_args()
    
    storage_resolver = StorageResolver()

    experiment_path = storage_resolver["experiments"].joinpath(args.experiment)
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
    for this_trial_name in args.trials:
        row = [this_trial_name]
        for target_name in experiment["targets"]:
            predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
                                            get_prediction_path(
                                                args.experiment,
                                                target_name,
                                                this_trial_name
                                            )), index_col="_id", squeeze=True)
            assert predictions.index.equals(true_targets.index)
            errors = np.abs(predictions - true_targets.loc[:, target_name])
            mae = errors.mean()
            mae_cv_std = np.std(errors.groupby(by=folds).mean())
            row.append(f"{mae*args.unit_multiplier:.1f} ± {mae_cv_std*args.unit_multiplier:.1f}")
        mae_table.add_row(row)
    print(mae_table)
            

if __name__ == "__main__":
    main()
