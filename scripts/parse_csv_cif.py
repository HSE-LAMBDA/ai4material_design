import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies, Element
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.cif import CifParser
import sys
sys.path.append('.')

from ai4mat.data.data import (
    get_dichalcogenides_innopolis,
    StorageResolver,
    Columns
)

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
    full_were_species = []
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
            full_were_species.append(reference_site.specie.Z)
            defect_energy_correction += single_atom_energies.loc[
                reference_site.specie, SINGLE_ENENRGY_COLUMN]
        # Substitution
        elif structure_dict[coords].specie != reference_site.specie:
            defects.append(structure_dict[coords])
            were_species.append(reference_site.specie.Z)
            full_were_species.append(reference_site.specie.Z)
            defect_energy_correction -= single_atom_energies.loc[
                structure_dict[coords].specie, SINGLE_ENENRGY_COLUMN]
            defect_energy_correction += single_atom_energies.loc[
                reference_site.specie, SINGLE_ENENRGY_COLUMN]
        else:
            full_were_species.append(reference_site.specie.Z)

    res = Structure(lattice=structure.lattice,
                    species=[x.specie for x in defects],
                    coords=[x.frac_coords for x in defects],
                    site_properties={"was": were_species},
                    coords_are_cartesian=False)
    res.state = [sorted([element.Z for element in reference_species])]


    structure_with_was = Structure(lattice=structure.lattice,
                    species=structure.species,
                    coords=structure.frac_coords,
                    site_properties={"was": full_were_species},
                    coords_are_cartesian=False)
    structure_with_was.state = [sorted([element.Z for element in reference_species])]
    return res, defect_energy_correction, structure_with_was


def main():
    parser = argparse.ArgumentParser("Parses csv/cif into pickle and targets.csv.gz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-folder", type=str)
    group.add_argument("--input-name", type=str)
    parser.add_argument("--populate-per-spin-target", action="store_true")
    args = parser.parse_args()

    storage_resolver = StorageResolver()
    if args.input_folder:
        dataset_name = Path(args.input_folder).name
        input_folder = args.input_folder
    else:
        dataset_name = args.input_name
        input_folder = storage_resolver["csv_cif"].joinpath(dataset_name)

    structures, defects = get_dichalcogenides_innopolis(input_folder)
    materials = defects.base.unique()
    unit_cells = {}
    for material in materials:
        unit_cells[material] = CifParser(Path(
            "defects_generation",
            "molecules",
            f"{material}.cif")).get_structures(primitive=False)[0]
    data_path = Path(input_folder)
    initial_structure_properties = pd.read_csv(
        data_path.joinpath("initial_structures.csv"),
        converters={"cell_size": lambda x: tuple(eval(x))},
        index_col=["base", "cell_size"])
    single_atom_energies = pd.read_csv(data_path.joinpath("elements.csv"),
                                       index_col="element",
                                       converters={"element": Element})

    COLUMNS = Columns()
    # TODO(kazeevn) this all is very ugly
    def get_defecs_from_row(row):
        defect_description = defects.loc[row[COLUMNS["structure"]["descriptor_id"]]]
        unit_cell = unit_cells[defect_description.base]
        initial_energy = initial_structure_properties.loc[
            (defect_description.base, defect_description.cell), "energy"].squeeze()
        defect_structure, formation_energy_part, structure_with_was = get_sparse_defect(
            row.initial_structure, unit_cell, defect_description.cell,
            single_atom_energies)
        return defect_structure, formation_energy_part + row.energy - initial_energy, structure_with_was

    defect_properties = structures.apply(get_defecs_from_row,
                                         axis=1,
                                         result_type="expand")
    structures = structures.drop(COLUMNS["structure"]["unrelaxed"], axis=1)                                     
    defect_properties.columns = [
        COLUMNS["structure"]["sparse_unrelaxed"],
        "formation_energy",
        COLUMNS["structure"]["unrelaxed"]
    ]
    structures = structures.join(defect_properties)
    structures["formation_energy_per_site"] = structures[
        "formation_energy"] / structures[COLUMNS["structure"]["sparse_unrelaxed"]].apply(len)
    structures["energy_per_atom"] = structures["energy"] / structures[COLUMNS["structure"]["unrelaxed"]].apply(len)

    assert structures.apply(lambda row: len(row[COLUMNS["structure"]["sparse_unrelaxed"]]) == len(
        defects.loc[row[COLUMNS["structure"]["descriptor_id"]], "defects"]), axis=1).all()
    
    if args.populate_per_spin_target:
        for kind in ("majority", "minority"):
            if f"band_gap_{kind}" in structures.columns:
                raise ValueError("Trying to set per-spin target, while they are already set")
            structures[f"band_gap_{kind}"] = structures["band_gap"]
            structures[f"homo_{kind}"] = structures["homo"]
            structures[f"lumo_{kind}"] = structures["lumo"]
        structures["total_mag"] = 0.

    save_dir = storage_resolver["processed"].joinpath(dataset_name)
    save_dir.mkdir(exist_ok=True, parents=True)
    structures.to_pickle(
        save_dir.joinpath("data.pickle.gz"))
    structures.drop(columns=[
            COLUMNS["structure"]["unrelaxed"],
            COLUMNS["structure"]["sparse_unrelaxed"]]
        ).to_csv(save_dir.joinpath("targets.csv.gz"),
            index_label=COLUMNS["structure"]["id"])


if __name__ == "__main__":
    main()
