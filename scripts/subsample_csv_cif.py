import argparse

from data import read_structures_descriptions, copy_indexed_structures

def main():
    parser = argparse.ArgumentParser("Subsamples csv/cif")
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Folder containing the input csv/cif dataset")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Folder to put the subsampled dataset")
    parser.add_argument("--sample-size", type=int, required=True,
                        help="The number of structures to sample")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--drop-na", action="store_true",
                        help="Drop the ids for which fields are missing in defects.csv")
    args = parser.parse_args()
    
    structures = read_structures_descriptions(args.input_folder)
    if args.drop_na:
        structures.dropna(inplace=True)
    structures = structures.sample(n=args.sample_size, random_state=args.random_state)
    copy_indexed_structures(structures, args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
