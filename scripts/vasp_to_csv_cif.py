import argparse
import sys
import numpy as np
from pathlib import Path
import shutil
from io import BytesIO
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

TARGET_FIELDS = ['energy', 'fermi_level', 'homo', 'lumo', 'band_gap']

def extract_data_from_vasp(vasprun_directory: Path) -> dict:
    """
    Extracts relevant fields from VASP output.
    """
    data = {}
    vasprun_output = vasprun_directory / '01_relax' / 'vasprun.xml'
    vasprun_file = Vasprun(vasprun_output)
    data['energy'] = vasprun_file.final_energy
    data['fermi_level'] = vasprun_file.efermi
    _, data['homo'], data['lumo'], _ = vasprun_file.eigenvalue_band_properties
    data['band_gap'] = vasprun_file.get_band_structure().get_band_gap()['energy']
    return data


def main():
    parser = argparse.ArgumentParser(description='Process raw VASP output into csv_cif dataset')
    parser.add_argument('--input-vasp', help='Directory containing VASP outputs. '
                        ' Usually found in datasets/raw_vasp', required=True)
    parser.add_argument('--input-metadata', help='Directory with defects.csv and descriptors.csv '
                        ' Usually found in datasets/POSCARs', required=True)
    parser.add_argument('--output-csv-cif', help='Output directory with csv+cif. '
                        'Usually shoudld be in datasts/csv_cif', required=True)
    parser.add_argument('--poscar-prefix', help='Prefix that makes POSCAR in '
                        'input-vasp from structure id', default="poscar_")
    args = parser.parse_args()

    structures_description = read_structures_descriptions(args.input_metadata)
    structures_description[TARGET_FIELDS] = np.nan
    input_VASP_dir = Path(args.input_vasp)
    output_csv_cif_dir = Path(args.output_csv_cif)
    output_csv_cif_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output_csv_cif_dir / 'initial.tar.gz', 'w:gz') as tar:
        for structure_id in tqdm(structures_description.index):
            structure_dir = input_VASP_dir / (args.poscar_prefix + structure_id)
            data = extract_data_from_vasp(structure_dir)
            structures_description.loc[structure_id, data.keys()] = data.values()
            structure = Poscar.from_file(structure_dir / "POSCAR").structure
            cif_string = str(CifWriter(structure)).encode('ASCII')
            tar_info = tarfile.TarInfo(name=structure_id + '.cif')
            tar_info.size=len(cif_string)
            tar.addfile(tar_info, BytesIO(cif_string))
    structures_description.to_csv(output_csv_cif_dir / 'defects.csv.gz')
    shutil.copyfile(Path(args.input_metadata) / 'descriptors.csv', output_csv_cif_dir / 'descriptors.csv')


if __name__ == "__main__":
    main()
