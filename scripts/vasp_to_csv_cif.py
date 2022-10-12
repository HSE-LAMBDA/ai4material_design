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
import numpy.typing as npt
sys.path.append('.')
from ai4mat.data.data import read_structures_descriptions, copy_indexed_structures


def get_E1(eigenvalues: dict[str, npt.NDArray], separate_spins: bool) -> list[float]:
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
    separate_spins: bool = False) -> dict[str, float]:
    """
    Extracts relevant fields from VASP output.
    """
    data = {}
    vasp_folder = vasprun_directory / '01_relax'
    vasprun_output = vasp_folder / 'vasprun.xml'
    # Here we delegate the occupancy tolerance default to pymatgen
    if band_occupancy_tolerence is None:
        vasprun_file = Vasprun(vasprun_output,
                            parse_potcar_file=False,
                            separate_spins=separate_spins,
                            parse_dos=True)
    else:
        vasprun_file = Vasprun(vasprun_output,
                            parse_potcar_file=False,
                            separate_spins=separate_spins,
                            occu_tol=band_occupancy_tolerence,
                            parse_dos=True)
    data['energy'] = vasprun_file.final_energy
    data['fermi_level'] = vasprun_file.efermi
    outcar = Outcar(vasp_folder / "OUTCAR")
    E_1 = get_E1(vasprun_file.eigenvalues, separate_spins)
    if separate_spins:
        assert len(E_1) == 2
        assert outcar.total_mag is not None
        data["total_mag"] = abs(outcar.total_mag)
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
    parser.add_argument('--input-vasp', type=Path, help='Directory containing VASP outputs. '
                        ' Usually found in datasets/raw_vasp', required=True)
    parser.add_argument('--input-structures-list', type=Path, help='Directory with defects.csv.gz, descriptors.csv '
                        'Usually found in datasets/POSCARs')
    intial_structures_source = parser.add_mutually_exclusive_group(required=True)
    intial_structures_source.add_argument('--POSCARs-in-input-list', action='store_true',
                                          help='Use the POSCARs in the --input-structures-list folder')
    intial_structures_source.add_argument('--input-structures-csv-cif', type=Path,
                                          help='Directory with a csv_cif dataset to copy initial structures from.')
    parser.add_argument('--vasprun-glob-prefix', help='Glob expression to get'
                        'vasp results folder from structure id', default="poscar_")
    parser.add_argument('--band-occupancy-tolerence', type=float,
                        help='Tolerence for band occupancy')
    parser.add_argument('--pristine-folder', type=str, help=
                        "Folder with chemical potentials (elements.csv) and "
                        "pristine structure energies (initial_structures.csv)")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing structures")
    parser.add_argument('--output-csv-cif', help='Output directory with csv+cif. '
                        'Usually should be in datasts/csv_cif', required=True)
    parser.add_argument("--separate-spins", action="store_true",
                        help="Report band gap separately for each spin channel")
    args = parser.parse_args()
    structures_description = read_structures_descriptions(args.input_structures_list)
    structures_list_path = Path(args.input_structures_list)
    input_VASP_dir = Path(args.input_vasp)
    output_csv_cif_dir = Path(args.output_csv_cif)
    output_csv_cif_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output_csv_cif_dir / 'initial.tar.gz', 'w:gz') as tar:    
        for structure_id in tqdm(structures_description.index):
            structure_dir_candidates = list(input_VASP_dir.glob(args.vasprun_glob_prefix + structure_id))
            if len(structure_dir_candidates) > 1:
                raise ValueError("More than one folder for structure id %s", structure_id)
            elif len(structure_dir_candidates) == 0:
                raise ValueError("No folder for structure id %s", structure_id)
            else:
                structure_dir = structure_dir_candidates[0]
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
            if args.POSCARs_in_input_list:
                structure = Poscar.from_file(structures_list_path / "poscars" / f"POSCAR_{structure_id}").structure
                cif_string = str(CifWriter(structure)).encode('ASCII')
                tar_info = tarfile.TarInfo(name=structure_id + '.cif')
                tar_info.size=len(cif_string)
                tar.addfile(tar_info, BytesIO(cif_string))
    if args.input_structures_csv_cif:
        copy_indexed_structures(structures_description.index,
                                args.intial_structures_source.input_structures_csv_cif / 'initial.tar.gz',
                                output_csv_cif_dir / 'initial.tar.gz')
    structures_description.to_csv(output_csv_cif_dir / 'defects.csv.gz')
    shutil.copyfile(structures_list_path / 'descriptors.csv', output_csv_cif_dir / 'descriptors.csv')
    pristine_path = Path(args.pristine_folder)    
    shutil.copyfile(pristine_path / 'elements.csv', output_csv_cif_dir / 'elements.csv')
    shutil.copyfile(pristine_path / 'initial_structures.csv', output_csv_cif_dir / 'initial_structures.csv')
    try:
        shutil.copytree(structures_list_path / 'unit_cells', output_csv_cif_dir / 'unit_cells')
    except FileNotFoundError:
        logging.warning("unit_cells not found in %s", structures_list_path / 'unit_cells')


if __name__ == "__main__":
    main()