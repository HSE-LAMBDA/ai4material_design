from pathlib import Path
import argparse
import pandas as pd
from pymatgen.io.vasp.outputs import Vasprun
from vasp_to_csv_cif import get_E1
from tqdm.auto import tqdm
import sys
sys.path.append('.')
from ai4mat.data.data import (
    Columns)

def read_E1(vasprun_path: Path) -> float:
    vasprun_file = Vasprun(vasprun_path,
                           parse_potcar_file=False,
                           separate_spins=False,
                           parse_dos=False)
    E_1 = get_E1(vasprun_file.eigenvalues, False)
    assert len(E_1) == 1
    return E_1[0]


def get_id_from_folder(folder: Path) -> str:
    split = folder.name.split("-")
    assert len(split) == 2
    return split[1]


def main():
    parser = argparse.ArgumentParser(description='Reads E_1 from VASP dataset, to be used for dichalcogenides8x8_vasp_nus_202110')
    parser.add_argument('--vasp', type=Path, help='Path to VASP output directory')
    parser.add_argument('--output', type=Path, help='Path to output file, expected to be a csv.gz file')
    args = parser.parse_args()
    E_1 = dict()
    for folder in tqdm(args.vasp.iterdir()):
        if not folder.is_dir():
            continue
        _id = get_id_from_folder(folder)
        E_1[_id] = read_E1(folder / "01_relax" / "vasprun.xml")
    E_1_pd = pd.Series(E_1, name="E_1")
    E_1_pd.to_csv(args.output, index_label=Columns()["structure"]["id"])


if __name__ == "__main__":
    main()