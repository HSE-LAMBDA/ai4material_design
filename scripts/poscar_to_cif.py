import argparse
from pathlib import Path
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.cif import CifWriter

def main():
    parser = argparse.ArgumentParser("Converts a folder with POSCARs to cif")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--poscar-prefix", type=str, default="poscar_S3-")
    args = parser.parse_args()
    
    processed = 0
    output_path = Path(args.output)
    for file_ in Path(args.input).iterdir():
        if file_.is_file() and file_.name.startswith(args.poscar_prefix):
            id_ = file_.name[len(args.poscar_prefix):]
            structure = Poscar.from_file(file_).structure
            CifWriter(structure).write_file(output_path.joinpath(f"{id_}.cif"))
            processed += 1
    print(f"Converted {processed} files")

if __name__ == "__main__":
    main()
