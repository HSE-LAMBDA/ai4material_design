import argparse
from pathlib import Path
import yaml
from itertools import combinations
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from ai4mat.data.data import (
    read_structures_descriptions,
    read_defects_descriptions,
    StorageResolver
)


def indices_intersect(dataframes):
    assert len(dataframes) == 2
    return len(dataframes[0].index.intersection(dataframes[1].index)) > 0


def get_folds(length, n_folds, random_state):
    fold_length_floor, remainder = divmod(length, n_folds)
    repetions = np.ones(n_folds, dtype=int) * fold_length_floor
    repetions[:remainder] += 1
    assert repetions.sum() == length
    return random_state.permutation(np.repeat(np.arange(n_folds), repetions))


def main():
    parser = argparse.ArgumentParser("Prepares CV data splits")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--targets", type=str, nargs="+",
                        default=["band_gap", "homo", "formation_energy_per_site"])
    parser.add_argument("--drop-na", action="store_true",
                        help="Drop the ids for which fields are missing")
    args = parser.parse_args()
    storage_resolver = StorageResolver()
    structures = [read_structures_descriptions(storage_resolver["csv_cif"]/dataset_name)
                  for dataset_name in args.datasets]
    if any(map(indices_intersect, combinations(structures, 2))):
        raise ValueError("Structures contain duplicate indices")
    structures = pd.concat(structures, axis=0)
    if args.drop_na:
        structures.dropna(inplace=True)

    defects = [read_defects_descriptions(storage_resolver["csv_cif"]/dataset_name)
               for dataset_name in args.datasets]
    if any(map(indices_intersect, combinations(defects, 2))):
        raise ValueError("Defects contain duplicate indices")
    defects = pd.concat(defects, axis=0)

    random_state = np.random.RandomState(args.random_seed)
    
    output_path = StorageResolver()["experiments"].joinpath(args.experiment_name)
    output_path.mkdir(exist_ok=True)
    fold_full = pd.Series(data=get_folds(len(structures), args.n_folds, random_state),
                          index=structures.index, name="fold")
    fold_full.to_csv(output_path.joinpath("folds.csv.gz"), index_label="_id")

    config = {
        "datasets": args.datasets,
        "strategy": "cv",
        "n-folds": args.n_folds,
        "targets": args.targets
    }
    with open(output_path.joinpath("config.yaml"), "wt") as config_file:
        yaml.dump(config, config_file)


if __name__ == "__main__":
    main()
