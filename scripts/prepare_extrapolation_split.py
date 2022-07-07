import argparse
import sys
import yaml
from itertools import combinations
import numpy as np
import pandas as pd

sys.path.append('.')
from prepare_data_split import indices_intersect

from ai4mat.data.data import (
    read_structures_descriptions,
    StorageResolver,
    TRAIN_FOLD,
    TEST_FOLD
)


def main():
    parser = argparse.ArgumentParser(
        "Prepares data splits for transfer learning validataion. "
        "1 - max(train fractions) of the train-test dataset is set aside as the test set. "
        "Then, it constructs two sets of experiments. "
        "The train set of the first one consists of the train-only datasets plus fractions of the"
        " train-test dataset. For the sectond set the training set consists only of "
        "the train-test dataset. The test set is the same for all experiments."
        "The train-test dataset splits are the same for all experiments with given fraction, and"
        "the greater train sets include the smaller train sets."
        )
    parser.add_argument("--train-only-datasets", type=str, nargs="+", required=True)
    parser.add_argument("--train-test-dataset", type=str, required=True)
    parser.add_argument("--train-fractions", nargs="+", type=float)
    parser.add_argument("--experiment-family-name", type=str, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--targets", type=str, nargs="+",
                        default=["band_gap", "formation_energy_per_site"])
    parser.add_argument("--drop-na", action="store_true",
                        help="Drop the ids for which fields are missing")
    args = parser.parse_args()

    train_only_structures = list(map(read_structures_descriptions, args.train_only_datasets))
    if any(map(indices_intersect, combinations(train_only_structures, 2))):
        raise ValueError("Structures contain duplicate indices")
    train_only_structures = pd.concat(train_only_structures, axis=0)

    train_test_structures = read_structures_descriptions(args.train_test_dataset)
    max_train_fraction = max(args.train_fractions)
    test_size = int((1. - max_train_fraction)*len(train_test_structures))
    if test_size <= 0:
        raise ValueError("No examples left for a test set.")

    rng = np.random.default_rng(args.random_seed)
    shuffled_train_test = rng.permutation(len(train_test_structures))

    test_indices = shuffled_train_test[:test_size]
    train_sizes = [int(fraction*len(train_test_structures)) for fraction in args.train_fractions]

    train_only_folds = pd.Series(
        data=np.full(len(train_only_structures), TRAIN_FOLD, dtype=np.int32),
        index=train_only_structures.index
    )

    test_fold = pd.Series(
        data=np.full(shape=len(test_indices), fill_value=TEST_FOLD, dtype=np.int32),
        index=train_test_structures.iloc[test_indices].index
    )

    output_path = StorageResolver()["experiments"].joinpath(args.experiment_family_name)
    output_path.mkdir(exist_ok=True)

    config = {
        "datasets": args.train_only_datasets + [args.train_test_dataset],
        "strategy": "train_test",
        "n-folds": 2,
        "targets": args.targets
    }

    family = []
    for train_size in train_sizes:
        this_size_path = output_path.joinpath(f"{train_size}")
        this_size_path.mkdir(exist_ok=True)

        train_indices = shuffled_train_test[test_size:test_size+train_size]
        in_domain_train_folds = pd.Series(
            data=np.full(len(train_indices), TRAIN_FOLD, np.int32),
            index=train_test_structures.iloc[train_indices].index
        )
        in_domain_folds = pd.concat([in_domain_train_folds, test_fold])
        if train_size > 0:
            in_domain_path = this_size_path.joinpath("in_domain")
            in_domain_path.mkdir(exist_ok=True)
            family.append(str(in_domain_path.relative_to(StorageResolver()["experiments"])))
            with open(in_domain_path.joinpath("config.yaml"), "wt") as config_file:
                yaml.dump(config, config_file)
            in_domain_folds.to_csv(in_domain_path.joinpath("folds.csv"), index_label="_id")
        
        all_folds = pd.concat([train_only_folds, in_domain_folds])
        out_domain_path = this_size_path.joinpath("in_and_out_domain")
        out_domain_path.mkdir(exist_ok=True)
        with open(out_domain_path.joinpath("config.yaml"), "wt") as config_file:
            yaml.dump(config, config_file)
        all_folds.to_csv(out_domain_path.joinpath("folds.csv"), index_label="_id")
        family.append(str(out_domain_path.relative_to(StorageResolver()["experiments"])))
    
    with open(output_path.joinpath("family.txt"), "wt") as family_file:
        family_file.write("\n".join(family))

if __name__ == "__main__":
    main()