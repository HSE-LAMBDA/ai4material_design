import argparse
from pathlib import Path
import pandas as pd
import sys
from tqdm import tqdm
import multiprocessing as mp

sys.path.append(".")

from scripts.structure_featurization import featurize
from ai4mat.data.data import get_dichalcogenides_innopolis, StorageResolver


def main():
    parser = argparse.ArgumentParser("Computes matminer features from csv/cif")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-folder", type=Path)
    group.add_argument("--input-name", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n-proc", type=int, default=1)
    args = parser.parse_args()

    storage_resolver = StorageResolver()
    if args.input_folder:
        raise NotImplementedError("Input folder")
        dataset_name = args.input_folder.name
        input_folder = args.input_folder
    else:
        dataset_name = args.input_name
        input_folder = storage_resolver["csv_cif"].joinpath(dataset_name)

    print("============== Data Loading ==============")
    structures, _ = get_dichalcogenides_innopolis(input_folder)
    if args.debug:
        structures = structures[:2]

    print("========== Features Computation ==========")
    with mp.Pool(args.n_proc) as pool:
        features = list(
            tqdm(
                pool.imap(featurize, structures["initial_structure"]),
                total=len(structures),
            )
        )
    features_df = pd.DataFrame(features, index=structures.index)

    save_dir = storage_resolver["processed"].joinpath(dataset_name)
    save_dir.mkdir(exist_ok=True)
    if args.debug:
        file_name = "matminer_dbg.csv.gz"
    else:
        file_name = "matminer.csv.gz"
    features_df.to_csv(save_dir / file_name)


if __name__ == "__main__":
    main()
