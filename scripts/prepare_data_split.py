import argparse
import yaml
from itertools import combinations
from functools import reduce
import pandas as pd
import numpy as np
import sys

sys.path.append('.')

from ai4mat.data.data import (
    read_structures_descriptions,
    StorageResolver,
    TEST_FOLD,
    VALIDATION_FOLD,
    TRAIN_FOLD,
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


def get_train_test_split(length, test_size, random_state):
    np.random.seed(random_state)
    res = np.zeros(length, dtype=int)
    amount_of_positives = int(length * test_size)
    res[:amount_of_positives] = TEST_FOLD
    return np.random.permutation(res)


def get_train_val_test_split(length, val_size, test_size, random_state):
    np.random.seed(random_state)
    val_length = int(length * val_size)
    test_length = int(length * test_size)
    res = np.zeros(length, dtype=int)
    res[:val_length] = VALIDATION_FOLD
    res[val_length: val_length + test_length] = TEST_FOLD
    return np.random.permutation(res)


def main():
    parser = argparse.ArgumentParser("Prepares CV data splits")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--validation-size", type=float)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--targets", type=str, nargs="+",
                        default=["homo_lumo_gap_min", "formation_energy_per_site"])
    parser.add_argument("--drop-na", action="store_true",
                        help="Drop the ids for which fields are missing")
    parser.add_argument("--strategy", type=str, default="cv")
    args = parser.parse_args()
    storage_resolver = StorageResolver()

    all_structures = [read_structures_descriptions(storage_resolver["csv_cif"] / dataset_name)
                      for dataset_name in args.datasets]
    average_len = reduce(lambda x, y: x + y, [i.shape[0] for i in all_structures])
    part_weight = average_len / len(all_structures)
    all_weights = [np.ones(len(i)) * part_weight / len(i) for i in all_structures]

    if any(map(indices_intersect, combinations(all_structures, 2))):
        raise ValueError("Structures contain duplicate indices")

    if args.strategy == "train_test":
        if args.test_size > 0:
            folds = np.concatenate([
                get_train_val_test_split(
                    all_structures[i].shape[0],
                    args.test_size,
                    args.validation_size,
                    args.random_seed
                ) for i in
                range(len(all_structures))
            ], axis=None)

            all_structures = pd.concat(all_structures, axis=0)
            all_weights = np.concatenate(all_weights, axis=None)
            if args.drop_na:
                all_structures.dropna(inplace=True)

            output_path = StorageResolver()["experiments"].joinpath(args.experiment_name + '_validation')
            output_path.mkdir(exist_ok=True)

            fold_full = pd.DataFrame(
                data={
                    'fold': folds,
                    'weight': all_weights
                },
                index=all_structures.index
            )

            fold_val = fold_full[fold_full['fold'] != TEST_FOLD].copy()
            fold_val['fold'] = np.where(fold_val['fold'] == VALIDATION_FOLD, TEST_FOLD, TRAIN_FOLD)
            fold_val.to_csv(output_path.joinpath('folds.csv.gz'), index_label='_id')

            config = {
                "datasets": args.datasets,
                "strategy": args.strategy,
                "n-folds": args.n_folds,
                "targets": args.targets
            }
            with open(output_path.joinpath("config.yaml"), "wt") as config_file:
                yaml.dump(config, config_file)

            output_path = StorageResolver()["experiments"].joinpath(args.experiment_name + '_test')
            output_path.mkdir(exist_ok=True)

            fold_val['fold'] = np.where(fold_val['fold'] == TEST_FOLD, TEST_FOLD, TRAIN_FOLD)
            fold_val.to_csv(output_path.joinpath('folds.csv.gz'), index_label='_id')

            config = {
                "datasets": args.datasets,
                "strategy": args.strategy,
                "n-folds": args.n_folds,
                "targets": args.targets
            }
            with open(output_path.joinpath("config.yaml"), "wt") as config_file:
                yaml.dump(config, config_file)
        else:
            raise NotImplementedError
    elif args.strategy == 'cv':
        random_state = np.random.RandomState(args.random_seed)

        output_path = StorageResolver()["experiments"].joinpath(args.experiment_name)
        output_path.mkdir(exist_ok=True)
        folds = get_folds(len(all_structures), args.n_folds, random_state)
        fold_full = pd.Series(data=folds,
                              index=all_structures.index, name="fold")
        fold_full.to_csv(output_path.joinpath("folds.csv.gz"), index_label="_id")

        config = {
            "datasets": args.datasets,
            "strategy": args.strategy,
            "n-folds": args.n_folds,
            "targets": args.targets
        }
        with open(output_path.joinpath("config.yaml"), "wt") as config_file:
            yaml.dump(config, config_file)


if __name__ == "__main__":
    main()
