import argparse
import os
from itertools import combinations
import pandas as pd
import numpy as np
from data import read_structures_descriptions, read_defects_descriptions


DEFAULT_DATA_ROOT = "datasets"


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
    parser = argparse.ArgumentParser("Prepares data splits for the paper")
    parser.add_argument("--datasets", nargs="+", default=[
        os.path.join(DEFAULT_DATA_ROOT, "dichalcogenides_innopolis_202105"),
        os.path.join(DEFAULT_DATA_ROOT, "dichalcogenides8x8_innopolis_202108")]
                        )
    parser.add_argument("--output-folder", type=str,
                        default=os.path.join(DEFAULT_DATA_ROOT,
                                             "paper_experiments",
                                             "inputs"))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=8)

    args = parser.parse_args()
    structures = list(map(read_structures_descriptions, args.datasets))
    if any(map(indices_intersect, combinations(structures, 2))):
        raise ValueError("Structures contain duplicate indices")
    structures = pd.concat(structures, axis=0).dropna()

    defects = list(map(read_defects_descriptions, args.datasets))
    if any(map(indices_intersect, combinations(defects, 2))):
        raise ValueError("Defects contain duplicate indices")
    defects = pd.concat(defects, axis=0)

    defects["vacancy_only"] = defects.defects.apply(lambda description: 
        all(map(lambda point_defect: point_defect["type"] == "vacancy", description)))

    random_state = np.random.RandomState(args.random_seed)
    fold_full = pd.Series(data=get_folds(len(structures), args.n_folds, random_state),
                          index=structures.index, name="fold")
    fold_full.to_csv(os.path.join(args.output_folder, "full.csv"), index_label="_id")

    is_vacancy_only = structures.descriptor_id.apply(lambda _id: defects.loc[_id, "vacancy_only"])
    vacancy_only = structures[is_vacancy_only]
    fold_vacancy_only = pd.Series(data=get_folds(len(vacancy_only), args.n_folds, random_state),
                                  index=vacancy_only.index, name="fold")
    fold_vacancy_only.to_csv(os.path.join(args.output_folder, "vacancy_only.csv"), index_label="_id")


if __name__ == "__main__":
    main()
