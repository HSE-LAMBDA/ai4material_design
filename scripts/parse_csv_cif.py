import argparse
from operator import methodcaller
from functools import cached_property
from pathlib import Path
import logging
import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import DummySpecies, Element
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN

import sys
sys.path.append('.')

from ai4mat.data.data import (
    get_dichalcogenides_innopolis,
    StorageResolver,
    Columns
)

class Shells:
    def __init__(self, structure):
        self.structure = structure
        self.nn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
  
    @cached_property
    def _all_nn_info(self):
        return self.nn.get_all_nn_info(self.structure)

    def get_nn_shell_info(self, site_idx, shell_idx):
        sites = self.nn._get_nn_shell_info(self.structure, self._all_nn_info, site_idx, shell_idx)
        output = []
        for info in sites:
            orig_site = self.structure[info["site_index"]]
            info["site"] = PeriodicSite(
                orig_site.species,
                np.add(orig_site.frac_coords, info["image"]),
                self.structure.lattice,
                properties=orig_site.properties,
            )
            output.append(info['site'])
        return output


class EOS:
    def __init__(self, num_shells=7) -> None:
        self.num_shells = num_shells

    @staticmethod
    def remove_other_species(structure, center):
        _struct = structure.copy()
        # _struct.remove_species(set(_struct.species).difference({center.spiece}))
        if (num_unique_species := len(set(structure.species))) == 1:
            return Structure.from_sites([site for site in structure if site.properties['center_index'] is not None] + [center])
        elif num_unique_species == 2:
            sites = [site for site in structure if site.specie != center.specie]
            sites.append(center)
            return Structure.from_sites(sites)
        else:
            raise NotImplementedError

    @staticmethod
    def get_distance_of_atoms_on_z_plane(center, sites):
        # return sorted({np.linalg.norm(site.coords - center.coords).round(3) for site in sites if site.coords[2].round(3) == center.coords[2].round(3)})
        return sorted({d for site in sites if (d := np.linalg.norm(site.coords[..., :2] - center.coords[..., :2]).round(3)) != 0})
        
    def get_shell(self, structure, site_idx, num_shells):
        shells = []
        # for i in range(1, num_shells):
        # shells.append(
        return self.shells_obj.get_nn_shell_info(site_idx, num_shells)
        # return shells

    def add_site_index_to_structure(self, structure):
        for i, site in enumerate(structure):
            site.properties['site_index'] = i
        return structure

    def find_center_index(self, structure, index):
        for i, site in enumerate(structure):
            if site.properties.get('center_index', -1) == index:
                return i
        
    def get_augmented_struct(self, structure):
        for center_idx, site in enumerate(structure):
            # add center index
            structure[center_idx].properties['center_index'] = center_idx 
            # get shells finder object
            self.shells_obj = Shells(structure)
            # get the shells
            shells_sites = self.get_shell(structure, self.find_center_index(structure, center_idx), self.num_shells) + [site]
            # remove all other species
            _struct = self.remove_other_species(Structure.from_sites(shells_sites), site)
            # add shells to the site
            site.properties['shells'] = self.get_distance_of_atoms_on_z_plane(site, [site for site in _struct])
            # delete the center index
            del site.properties['center_index']

        assert all(map(lambda s: 'shells' in s.properties, structure))
        return structure


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
    full_were_species = []
    defect_energy_correction = 0

    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)

    for coords, reference_site in reference_structure_dict.items():
        # Vacancy
        reference_site.properties['was'] =  reference_site.specie.Z

        if coords not in structure_dict:
            defects.append(
                PeriodicSite(
                    species=DummySpecies(),
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                    properties=reference_site.properties,
                ))
            full_were_species.append(reference_site.specie.Z)
            defect_energy_correction += single_atom_energies.loc[
                reference_site.specie, SINGLE_ENENRGY_COLUMN]
        # Substitution
        elif structure_dict[coords].specie != reference_site.specie:
            structure_dict[coords].properties = reference_site.properties 
            defects.append(structure_dict[coords])
            
            full_were_species.append(reference_site.specie.Z)
            defect_energy_correction -= single_atom_energies.loc[
                structure_dict[coords].specie, SINGLE_ENENRGY_COLUMN]
            defect_energy_correction += single_atom_energies.loc[
                reference_site.specie, SINGLE_ENENRGY_COLUMN]
        else:
            full_were_species.append(reference_site.specie.Z)

    res = Structure.from_sites(defects)
    res.state = [sorted([element.Z for element in reference_species])]

    # left for legacy support
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
    parser.add_argument("--fill-missing-band-properties", action="store_true")
    parser.add_argument("--normalize-homo-lumo", action="store_true")
    parser.add_argument("--skip-eos", action="store_true",
                        help="Don't add EOS indices")
    args = parser.parse_args()

    storage_resolver = StorageResolver()
    if args.input_folder:
        dataset_name = Path(args.input_folder).name
        input_folder = args.input_folder
    else:
        dataset_name = args.input_name
        input_folder = storage_resolver["csv_cif"].joinpath(dataset_name)

    structures, defects = get_dichalcogenides_innopolis(input_folder)
    if (args.fill_missing_band_properties and
        "homo_lumo_gap" not in structures.columns and
        "homo" in structures.columns and
        "lumo" in structures.columns):
        structures["homo_lumo_gap"] = structures["lumo"] - structures["homo"]
    materials = defects.base.unique()
    unit_cells = {}
    for material in materials:
        try:
            unit_cells[material] = CifParser(
                input_folder/"unit_cells"/f"{material}.cif").get_structures(primitive=False)[0]
        except FileNotFoundError:
            logging.warning(f"Unit cell for {material} not found in the dataset folder, using the global one")
            unit_cells[material] = CifParser(Path(
                "defects_generation",
                "molecules",
                f"{material}.cif")).get_structures(primitive=False)[0]
        if not args.skip_eos:
            unit_cells[material] = EOS().get_augmented_struct(unit_cells[material])

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
        initial_energy = initial_structure_properties.at[
            (defect_description.base, defect_description.cell), "energy"]
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


    if args.normalize_homo_lumo:
        for kind in (None, "majority", "minority"):
            for property in ("homo", "lumo"):
                if kind is None:
                    column = property
                else:
                    column = f"{property}_{kind}"
                if column not in structures.columns:
                    continue
                defects_per_structure = defects.loc[structures[COLUMNS["structure"]["descriptor_id"]]]
                defects_key = (defects_per_structure.base.unique(), defects_per_structure.cell.unique())
                if len(defects_key[0]) != 1 or len(defects_key[1]) != 1:
                    raise NotImplementedError("Handling different pristine materials in same dataset not implemented")
                defects_key = (defects_key[0][0], defects_key[1][0])
                normalization_constant = initial_structure_properties.at[defects_key, "E_VBM"] - initial_structure_properties.at[defects_key, "E_1"]
                structures[f"normalized_{column}"] = \
                    structures[column] - structures["_".join(filter(None, ("E_1", kind)))] - normalization_constant


    for property in ("homo_lumo_gap", "homo", "lumo", "normalized_homo", "normalized_lumo"):
        for kind in ("min", "max"):
            column = f"{property}_{kind}"
            if column in structures.columns:
                raise ValueError(f"Column {column} already exists, it's not supposed to at this stage")
            source_columns = [f"{property}_majority", f"{property}_minority"]
            if not frozenset(source_columns).issubset(structures.columns):
                logging.info("Skipped filling %s, as %s_{majority,minority} are not available", column, property)
                continue
            structures[column] = methodcaller(kind, axis=1)(structures.loc[:, source_columns])


    if args.fill_missing_band_properties:
        for kind in ("majority", "minority", "max", "min"):
            for property in ("homo_lumo_gap", "homo", "lumo", "normalized_homo", "normalized_lumo"):
                spin_column = f"{property}_{kind}"
                if spin_column not in structures.columns:
                    if property in structures.columns:
                        structures[spin_column] = structures[property]
                        logging.info("Filling {}", spin_column)
                    else:
                        logging.warning(r"%s is missing in data, can't fill %s", property, spin_column)
        if "total_mag" not in structures.columns:
            structures["total_mag"] = 0.
            logging.info("Setting total_mag = 0")

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
