import argparse
import shutil
from pathlib import Path

from data import read_structures_descriptions

def main():
    parser = argparse.ArgumentParser("Subsamples csv/cif")
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--drop-missing", action="store_true")
    args = parser.parse_args()
    
    structures = read_structures_descriptions(args.input_folder)
    if args.drop_missing:
        structures.dropna(inplace=True)
    structures = structures.sample(n=args.sample_size, random_state=args.random_state)

    save_path = Path(args.output_folder)
    save_path.mkdir(parents=True)
    # since we don't clean, raise if output exists
    for file_name in ("descriptors.csv", "elements.csv", "initial_structures.csv"):
        shutil.copy2(Path(args.input_folder, file_name),
                     save_path.joinpath(file_name))
    structures_folder = save_path.joinpath("initial")
    structures_folder.mkdir()
    input_structures_folder = Path(args.input_folder, "initial")
    for structure_id in structures.index:
        file_name = f"{structure_id}.cif"
        shutil.copy2(input_structures_folder.joinpath(file_name),
                     structures_folder.joinpath(file_name))
    structures.to_csv(save_path.joinpath("defects.csv"),
                      index_label="_id")


if __name__ == "__main__":
    main()
