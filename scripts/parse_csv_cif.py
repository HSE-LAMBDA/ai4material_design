import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies, Element
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.cif import CifParser

from data import get_dichalcogenides_innopolis

SINGLE_ENENRGY_COLUMN = "chemical_potential"


def get_frac_coords_set(structure):
    return set(map(tuple, np.round(structure.frac_coords, 3)))


def strucure_to_dict(structure, precision=3):
    res = {}
    for site in structure:
        res[tuple(np.round(site.frac_coords, precision))] = site
    return res


def get_sparse_defect(structure, unit_cell, supercell_size,
                      single_atom_energies):
    reference_species = set(unit_cell.species)
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    reference_sites = get_frac_coords_set(reference_supercell)

    defects = []
    were_species = []
    defect_energy_correction = 0

    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)

    for coords, reference_site in reference_structure_dict.items():
        # Vacancy
        if coords not in structure_dict:
            defects.append(
                PeriodicSite(
                    species=DummySpecies(),
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                ))
            were_species.append(reference_site.specie.Z)
            defect_energy_correction += single_atom_energies.loc[
                reference_site.specie, SINGLE_ENENRGY_COLUMN]
        # Substitution
        elif structure_dict[coords].specie != reference_site.specie:
            defects.append(structure_dict[coords])
            were_species.append(reference_site.specie.Z)
            defect_energy_correction -= single_atom_energies.loc[
                structure_dict[coords].specie, SINGLE_ENENRGY_COLUMN]
            defect_energy_correction += single_atom_energies.loc[
                reference_site.specie, SINGLE_ENENRGY_COLUMN]

    res = Structure(lattice=structure.lattice,
                    species=[x.specie for x in defects],
                    coords=[x.frac_coords for x in defects],
                    site_properties={"was": were_species},
                    coords_are_cartesian=False)
    res.state = [sorted([element.Z for element in reference_species])]
    return res, defect_energy_correction


def main():
    parser = argparse.ArgumentParser("Parses csv/cif into pickle and targets.csv")
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--ignore-missing", action="store_true")
    args = parser.parse_args()

    structures, defects = get_dichalcogenides_innopolis(args.input_folder)
    materials = defects.base.unique()
    unit_cells = {}
    for material in materials:
        unit_cells[material] = CifParser(Path().joinpath(
            "defects_generation", "molecules",
            f"{material}.cif")).get_structures(primitive=True)[0]
    data_path = Path(args.input_folder)
    initial_structure_properties = pd.read_csv(
        data_path.joinpath("initial_structures.csv"),
        index_col=["base", "cell_length"],
        usecols=[1, 2, 3, 4])
    single_atom_energies = pd.read_csv(data_path.joinpath("elements.csv"),
                                       index_col=0,
                                       converters={0: Element})

    # TODO(kazeevn) this all is very ugly
    def get_defecs_from_row(row):
        try:
            defect_description = defects.loc[row.descriptor_id]
            unit_cell = unit_cells[defect_description.base]
            initial_energy = initial_structure_properties.loc[
                defect_description.base, defect_description.cell[0]].energy
            defect_structure, formation_energy_part = get_sparse_defect(
                row.initial_structure, unit_cell, defect_description.cell,
                single_atom_energies)
            return defect_structure, formation_energy_part + row.energy - initial_energy
        except KeyError:
            if args.ignore_missing:
                return None, None
            else:
                raise

    defect_properties = structures.apply(get_defecs_from_row,
                                         axis=1,
                                         result_type="expand")
    defect_properties.columns = ["defect_representation", "formation_energy"]
    structures = structures.join(defect_properties)
    if args.ignore_missing:
        structures = structures.dropna()
    structures["formation_energy_per_site"] = structures[
        "formation_energy"] / structures["defect_representation"].apply(len)
    structures["band_gap"] = structures["lumo"] - structures["homo"]

    # assert structures.apply(lambda row: len(row.defect_representation) == len(
    #    defects.loc[row.descriptor_id, "defects"]),
    #                        axis=1).all()

    save_dir = Path(args.output_folder)
    save_dir.mkdir(exist_ok=True)
    structures.to_pickle(
        save_dir.joinpath("data.pickle.gzip"))
    structures.to_csv(save_dir.joinpath("targets.csv"),
                      index_label="_id",
                      columns=[
                          "energy", "energy_per_atom", "formation_energy",
                          "formation_energy_per_site", "band_gap", "homo",
                          "lumo", "fermi_level"
                      ])


if __name__ == "__main__":
    main()
