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
from pymatgen.io.cif import CifWriter
sys.path.append('.')
from ai4mat.data.data import read_structures_descriptions


def extract_data_from_vasp(
    vasprun_directory: Path,
    band_occupancy_tolerence: float = None,
    separate_spins: bool = False) -> dict:
    """
    Extracts relevant fields from VASP output.
    """
    data = {}
    vasprun_output = vasprun_directory / '01_relax' / 'vasprun.xml'
    vasprun_file = Vasprun(vasprun_output,
                           parse_potcar_file=False,
                           occu_tol=band_occupancy_tolerence,
                           separate_spins=separate_spins,
                           parse_dos=True)
    data['energy'] = vasprun_file.final_energy
    data['fermi_level'] = vasprun_file.efermi
    if separate_spins:
         [data['band_gap_1'], data['band_gap_2']], \
         [data['homo_1'], data['homo_2']], \
         [data['lumo_1'], data['lumo_2']], \
         [data["is_band_gap_direct_1"], data["is_band_gap_direct_2"]] = \
            vasprun_file.eigenvalue_band_properties
    else:
        data['band_gap'],\
            data['homo'],\
            data['lumo'],\
            data["is_band_gap_direct"] = \
            vasprun_file.eigenvalue_band_properties
    return data


def make_1_2(list_):
    result = []
    for item in list_:
        result.append(item + "_1")
        result.append(item + "_2")
    return result


def main():
    parser = argparse.ArgumentParser(description='Process raw VASP output into csv_cif dataset')
    parser.add_argument('--input-vasp', help='Directory containing VASP outputs. '
                        ' Usually found in datasets/raw_vasp', required=True)
    parser.add_argument('--input-structures', help='Directory with defects.csv, descriptors.csv '
                        'and initial structures POSCARs. Usually found in datasets/POSCARs', required=True)
    parser.add_argument('--output-csv-cif', help='Output directory with csv+cif. '
                        'Usually shoudld be in datasts/csv_cif', required=True)
    parser.add_argument('--poscar-prefix', help='Prefix that makes POSCAR in '
                        'input-vasp from structure id', default="poscar_")
    parser.add_argument('--band-occupancy-tolerence', type=float, default=1e-8,
                        help='Tolerence for band occupancy')
    # Default from pymatgen v2022.7.25
    # https://github.com/materialsproject/pymatgen/blob/baf62b77788fc43387e15b1e0b60094132815c47/pymatgen/io/vasp/outputs.py#L313
    parser.add_argument("--separate-spins", action="store_true",
                        help="Report band gap separately for each spin channel")
    args = parser.parse_args()
    structures_description = read_structures_descriptions(args.input_structures)
    structures_path = Path(args.input_structures)
    TARGET_FIELDS = [
        'energy',
        'fermi_level']
    band_targets = [
            'homo',
            'lumo',
            'band_gap',
            'is_band_gap_direct'
        ]
    if args.separate_spins:
        TARGET_FIELDS += make_1_2(band_targets)
    else:
        TARGET_FIELDS += band_targets

    structures_description[TARGET_FIELDS] = np.nan
    input_VASP_dir = Path(args.input_vasp)
    output_csv_cif_dir = Path(args.output_csv_cif)
    output_csv_cif_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output_csv_cif_dir / 'initial.tar.gz', 'w:gz') as tar:    
        for structure_id in tqdm(structures_description.index):
            structure_dir = input_VASP_dir / (args.poscar_prefix + structure_id)
            data = extract_data_from_vasp(structure_dir,
                                          band_occupancy_tolerence=args.band_occupancy_tolerence,
                                          separate_spins=args.separate_spins)
            structures_description.loc[structure_id, data.keys()] = data.values()
            structure = Poscar.from_file(structures_path / "poscars" / f"POSCAR_{structure_id}").structure
            cif_string = str(CifWriter(structure)).encode('ASCII')
            tar_info = tarfile.TarInfo(name=structure_id + '.cif')
            tar_info.size=len(cif_string)
            tar.addfile(tar_info, BytesIO(cif_string))
    structures_description.to_csv(output_csv_cif_dir / 'defects.csv.gz')
    shutil.copyfile(structures_path / 'descriptors.csv', output_csv_cif_dir / 'descriptors.csv')


if __name__ == "__main__":
    main()
