import pandas as pd
from argparse import ArgumentParser
import yaml

import sys
sys.path.append('.')
from ai4mat.data.data import (
    read_structures_descriptions,
    StorageResolver,
    TRAIN_FOLD,
    TEST_FOLD
)


def main():
    parser = ArgumentParser("Prepares experiment folds from a config file")
    parser.add_argument("--experiment-name", type=str, required=True)

    args = parser.parse_args()
    storage_resolver = StorageResolver()
    output_path = storage_resolver["experiments"].joinpath(args.experiment_name)
    with open(output_path.joinpath("config.yaml"), "rt") as config_file:
        config = yaml.safe_load(config_file)
    
    assert set(config["datasets"]) == set(config["train"]).union(config["test"])
    assert len(set(config["test"]).intersection(config["train"])) == 0

    train_structures = pd.concat([read_structures_descriptions(storage_resolver["csv_cif"] / dataset_name) for dataset_name in config["train"]], axis=0)
    train_folds = pd.Series(data=TRAIN_FOLD, index=train_structures.index, name="fold", dtype=int)
    test_structures = pd.concat([read_structures_descriptions(storage_resolver["csv_cif"] / dataset_name) for dataset_name in config["test"]], axis=0)
    test_folds = pd.Series(data=TEST_FOLD, index=test_structures.index, name="fold", dtype=int)
    
    folds = pd.concat([train_folds, test_folds], axis=0)
    folds_path = output_path.joinpath("folds.csv.gz")
    if folds_path.exists():
        raise ValueError("Refusing to overrite an existing experiment")
    folds.to_csv(folds_path, index_label="_id")


if __name__ == "__main__":
    main()

