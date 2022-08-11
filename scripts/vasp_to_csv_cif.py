import argparse
import sys
import numpy as np
from pathlib import Path
import shutil
from io import StringIO
from tqdm.auto import tqdm
import tarfile
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Poscar
sys.path.append('.')
from pymatgen.io.cif import CifWriter
from ai4mat.data.data import (
    StorageResolver,
    read_structures_descriptions
)

TARGET_FIELDS = ['energy', 'final_structure', 'fermi_level', 'homo', 'lumo', 'band_gap']

def extract_data_from_vasp(vasprun_directory: Path) -> dict:
    data = {}
    vasprun_output = vasprun_directory / '01_relax' / 'vasprun.xml'
    vasprun_file = Vasprun(vasprun_output)
    data['energy'] = vasprun_file.final_energy
    data['final_structure'] = vasprun_file.final_structure
    data['fermi_level'] = vasprun_file.efermi
    _, data['homo'], data['lumo'], _ = vasprun_file.eigenvalue_band_properties
    data['band_gap'] = vasprun_file.get_band_structure().get_band_gap()['energy']
    return data

def main():
    parser = argparse.ArgumentParser(description='Process raw VASP output into csv_cif dataset')
    parser.add_argument('--input-vasp', help='Directory containing VASP outputs. '
                        ' Usually found in datasets/raw_vasp', required=True)
    parser.add_argument('--input-metadata', help='Directory with defects.csv and descriptors.csv. '
                        ' Usually found in datasets/POSCARs', required=True)
    parser.add_argument('--output-csv-cif', help='Output directory with csv+cif. '
                        'Usually shoudld be in datasts/csv_cif', required=True)
    parser.add_argument('--poscar-prefix', help='Prefix that from structured ids '
                        'makes POSCARs in input-vasp', default="poscar_")
    args = parser.parse_args()

    structures_description = read_structures_descriptions(args.input_metadata)
    structures_description[TARGET_FIELDS] = np.nan
    input_VASP_dir = Path(args.input_vasp)
    output_csv_cif_dir = Path(args.output_csv_cif)
    with tarfile.open(output_csv_cif_dir / 'initial.tar.gz', 'w:gz') as tar:
        for structure_id in tqdm(structures_description.index):
            structure_dir = input_VASP_dir / args.poscar_prefix + structure_id
            data = extract_data_from_vasp(structure_dir)
            structures_description.loc[structure_id, data.keys()] = data.values()
            structure = Poscar.from_file(structure_dir / "POSCAR").structure
            cif_file = StringIO()
            CifWriter(structure).write_file(cif_file)
            tar.addfile(tarfile.TarInfo(f'{structure_id}.cif'), cif_file)
    structures_description.to_csv(output_csv_cif_dir / 'defects.csv')
    shutil.copyfile(args.input_metadata / 'descriptors.csv', output_csv_cif_dir / 'descriptors.csv')

if __name__ == "__main__":
    main()
