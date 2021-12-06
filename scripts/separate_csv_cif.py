import argparse
import numpy as np
import pandas as pd
from data import (read_structures_descriptions,
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
    selected_defects = defects[selection]
    structures = structures.loc[structures.descriptor_id.isin(selected_defects.index)]
    copy_indexed_structures(structures, args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
