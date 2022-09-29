import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from io import BytesIO
from tqdm.auto import tqdm
import tarfile
import logging
from xml.etree.ElementTree import ParseError
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.cif import CifWriter
sys.path.append('.')
from ai4mat.data.data import read_structures_descriptions, read_defects_descriptions


def get_E1(eigenvalues: dict[str, np.ndarray], separate_spins: bool) -> list[float]:
    """
    Extracts the energy of the lowest orbital from vasprun.xml.
    Args:
        eigenvalues: dictionary of eigenvalues from vasprun.xml,
            vasprun_file.eigenvalues
        separate_spins: whether to separate spins
    Returns:
        E1: list of E1 values
    """
    if separate_spins:
        if len(eigenvalues) != 2:
            raise ValueError("Separate spins requires 2 spin channels")
    else:
        if len(eigenvalues) != 1:
            raise NotImplemented("Collating values for different spins is not implemented")
    result = []
    for d in eigenvalues.values():
            if d.shape[0] != 1:
                raise NotImplemented("Multiple k-point not implemented")
            result.append(d[0, 0, 0])
    return result


def extract_data_from_vasp(
    vasprun_directory: Path,
    band_occupancy_tolerence: float = None,
    separate_spins: bool = False) -> dict:
    """
    Extracts relevant fields from VASP output.
    """
    data = {}
    vasp_folder = vasprun_directory / '01_relax'
    vasprun_output = vasp_folder / 'vasprun.xml'
    vasprun_file = Vasprun(vasprun_output,
                           parse_potcar_file=False,
                           occu_tol=band_occupancy_tolerence,
                           separate_spins=separate_spins,
                           parse_dos=True)
    data['energy'] = vasprun_file.final_energy
    data['fermi_level'] = vasprun_file.efermi
    outcar = Outcar(vasp_folder / "OUTCAR")
    E_1 = get_E1(vasprun_file.eigenvalues, separate_spins)
    if separate_spins:
        assert len(E_1) == 2
        data["total_mag"] = np.abs(outcar.total_mag)
        eigenvalue_band_properties = list(zip(*vasprun_file.eigenvalue_band_properties))
        indices = {
            "majority": int(outcar.total_mag < 0),
            "minority": int(outcar.total_mag >= 0)}
        for kind, index in indices.items():
            data[f'band_gap_{kind}'], \
                data[f'lumo_{kind}'], \
                data[f'homo_{kind}'], _ = eigenvalue_band_properties[index]
            data[f'E_1_{kind}'] = E_1[index]
    else:
        data['band_gap'],\
            data['lumo'],\
            data['homo'], _ = \
            vasprun_file.eigenvalue_band_properties
        assert len(E_1) == 1
        data["E_1"] = E_1[0]
    return data


def main():
    parser = argparse.ArgumentParser(description='Process raw VASP output into csv_cif dataset')
    parser.add_argument('--input-vasp', help='Directory containing VASP outputs. '
                        ' Usually found in datasets/raw_vasp', required=True)
    parser.add_argument('--input-structures', help='Directory with defects.csv.gz, descriptors.csv '
                        'and initial structures POSCARs. Usually found in datasets/POSCARs', required=True)
    parser.add_argument('--output-csv-cif', help='Output directory with csv+cif. '
                        'Usually should be in datasts/csv_cif', required=True)
    parser.add_argument('--poscar-prefix', help='Prefix that makes POSCAR in '
                        'input-vasp from structure id', default="poscar_")
    parser.add_argument('--band-occupancy-tolerence', type=float, default=5e-2,
                        help='Tolerence for band occupancy')
    parser.add_argument('--pristine-folder', type=str, help=
                        "Folder with chemical potentials (elements.csv) and "
                        "pristine structure energies (initial_structures.csv)")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing structures")
    # Default from pymatgen v2022.7.25
    # https://github.com/materialsproject/pymatgen/blob/baf62b77788fc43387e15b1e0b60094132815c47/pymatgen/io/vasp/outputs.py#L313
    parser.add_argument("--separate-spins", action="store_true",
                        help="Report band gap separately for each spin channel")
    args = parser.parse_args()
    structures_description = read_structures_descriptions(args.input_structures)
    structures_path = Path(args.input_structures)
    input_VASP_dir = Path(args.input_vasp)
    output_csv_cif_dir = Path(args.output_csv_cif)
    output_csv_cif_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output_csv_cif_dir / 'initial.tar.gz', 'w:gz') as tar:    
        for structure_id in tqdm(structures_description.index):
            structure_dir = input_VASP_dir / (args.poscar_prefix + structure_id)
            try:
                data = extract_data_from_vasp(structure_dir,
                                            band_occupancy_tolerence=args.band_occupancy_tolerence,
                                            separate_spins=args.separate_spins)
            except (FileNotFoundError, ParseError) as e:
                if args.allow_missing:
                    if isinstance(e, FileNotFoundError):
                        logging.warning("VASP data for %s not found", structure_id)
                    else:
                        logging.warning("VASP data for %s is corrupted", structure_id)
                    continue
                else:
                    raise e
            structures_description.loc[structure_id, data.keys()] = data.values()
            structure = Poscar.from_file(structures_path / "poscars" / f"POSCAR_{structure_id}").structure
            cif_string = str(CifWriter(structure)).encode('ASCII')
            tar_info = tarfile.TarInfo(name=structure_id + '.cif')
            tar_info.size=len(cif_string)
            tar.addfile(tar_info, BytesIO(cif_string))

    structures_description.to_csv(output_csv_cif_dir / 'defects.csv.gz')
    shutil.copyfile(structures_path / 'descriptors.csv', output_csv_cif_dir / 'descriptors.csv')
    pristine_path = Path(args.pristine_folder)    
    shutil.copyfile(pristine_path / 'elements.csv', output_csv_cif_dir / 'elements.csv')
    shutil.copyfile(pristine_path / 'initial_structures.csv', output_csv_cif_dir / 'initial_structures.csv')
    try:
        shutil.copytree(structures_path / 'unit_cells', output_csv_cif_dir / 'unit_cells')
    except FileNotFoundError:
        logging.warning("unit_cells not found in %s", structures_path / 'unit_cells')


if __name__ == "__main__":
    main()