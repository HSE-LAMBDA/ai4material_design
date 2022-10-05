import argparse
import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path
import shutil
sys.path.append('.')
from ai4mat.data.data import (
    read_structures_descriptions,
    read_defects_descriptions,
    copy_indexed_structures)


def main():
    parser = argparse.ArgumentParser("Samples from csv/cif by base material"
                                     " and supercell size.")
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Folder containing the input csv/cif dataset")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Folder to put the sampled dataset")
    parser.add_argument("--drop-na", action="store_true",
                        help="Drop the ids for which fields are missing in defects.csv")
    parser.add_argument("--base-material", type=str,
                        help="Base material, e. g. MoS2")
    parser.add_argument("--supercell-size", type=int,
                        help="Component 0 of the supercell shape.")
    parser.add_argument("--vacancy-only", action="store_true")

    args = parser.parse_args()
    
    structures = read_structures_descriptions(args.input_folder)
    if args.drop_na:
        structures.dropna(inplace=True)

    defects = read_defects_descriptions(args.input_folder)
    selection = pd.Series(
        index=defects.index,
        data=np.ones(len(defects), dtype=bool))
    if args.base_material:
        selection = selection & (defects.base == args.base_material)
    if args.supercell_size:
        selection = selection & (defects.cell.apply(lambda l: l[0]) == args.supercell_size)
    if args.vacancy_only:
        selection = selection & (defects.defects.apply(lambda defect: all((this_defect['type'] == 'vacancy' for this_defect in defect))))
    selected_defects = defects[selection]
    structures = structures.loc[structures.descriptor_id.isin(selected_defects.index)]

    save_path = Path(args.output_folder)
    save_path.mkdir(parents=True)
    # since we don't clean, raise if output exists
    for file_name in ("elements.csv", "initial_structures.csv"):
        shutil.copy2(Path(args.input_folder, file_name),
                     save_path.joinpath(file_name))
    output_structures = save_path.joinpath("initial.tar.gz")
    input_path = Path(args.input_folder)
    input_structures = input_path / "initial.tar.gz"
    copy_indexed_structures(structures.index, input_structures, output_structures)
    structures.to_csv(save_path.joinpath("defects.csv.gz"),
                      index_label="_id")
    selected_defects.to_csv(save_path.joinpath("descriptors.csv"), index_label="_id")
    try:
        shutil.copytree(input_path / 'unit_cells', save_path / 'unit_cells')
    except FileNotFoundError:
        logging.warning("unit_cells not found in %s", input_path / 'unit_cells')


if __name__ == "__main__":
    main()
