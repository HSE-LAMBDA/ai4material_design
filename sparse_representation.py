import os
import pandas as pd
import numpy as np
from data import get_dichalcogenides_innopolis

from pymatgen.core.sites import PeriodicSite
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies, Element
from pymatgen.io.cif import CifParser

SINGLE_ENENRGY_COLUMN = "energy_bulk_estimate"


def get_frac_coords_set(structure):
  return set(map(tuple, np.round(structure.frac_coords, 3)))


def strucure_to_dict(structure, precision=3):
  res = {}
  for site in structure:
    res[tuple(np.round(site.frac_coords, precision))] = site
  return res


def get_sparse_defect(structure, unit_cell, supercell_size, single_atom_energies):
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
      defects.append(PeriodicSite(
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
    structures_small, defects_small = get_dichalcogenides_innopolis("datasets/dichalcogenides_innopolis_202105/")
    structures_8x8, defects_8x8 = get_dichalcogenides_innopolis("datasets/dichalcogenides8x8_innopolis_202108/")
    assert len(structures_8x8.index.intersection(structures_small.index)) == 0
    defects = pd.concat([defects_small, defects_8x8], axis=0)
    materials = defects.base.unique()
    unit_cells = {}
    for material in materials:
        unit_cells[material] = CifParser(os.path.join(
            "defects_generation", "molecules", f"{material}.cif")).get_structures(primitive=True)[0]
    initial_structure_properties = pd.concat([
        pd.read_csv(os.path.join("datasets", "dichalcogenides_innopolis_202105", "initial_structures.csv"),
                    index_col=["base", "cell_length"], usecols=[1,2,3,4]),
        pd.read_csv(os.path.join("datasets", "dichalcogenides8x8_innopolis_202108", "initial_structures.csv"),
                    index_col=["base", "cell_length"], usecols=[1,2,3,4])
    ], axis=0)
    single_atom_energies = pd.read_csv(os.path.join("datasets", "single_atom_energies.csv"),
                                   index_col=0,
                                   converters={0: Element})
    temporary_energies = {"Mo": -10.85, "W": -12.96, "S": -4.24, "Se": -3.50, "O": -4.95}
    for element_name, energy in temporary_energies.items():
        single_atom_energies.loc[Element(element_name), SINGLE_ENENRGY_COLUMN] = energy

    # TODO(kazeevn) this all is very ugly
    def get_defecs_from_row(row):
        defect_description = defects.loc[row.descriptor_id]
        unit_cell = unit_cells[defect_description.base]
        initial_energy = initial_structure_properties.loc[defect_description.base, defect_description.cell[0]].energy
        defect_structure, formation_energy_part = get_sparse_defect(
            row.initial_structure,
            unit_cell,
            defect_description.cell,
            single_atom_energies)
        return defect_structure, formation_energy_part + row.energy - initial_energy

    all_structures = pd.concat([structures_small, structures_8x8.dropna()], axis=0)
    defect_properties = all_structures.apply(
        get_defecs_from_row,
        axis=1,
        result_type="expand")
    defect_properties.columns = ["defect_representation", "formation_energy"]
    all_structures = all_structures.join(defect_properties)

    all_structures["formation_energy_per_site"] = all_structures["formation_energy"] / all_structures["defect_representation"].apply(len)
    all_structures["band_gap"] = all_structures["lumo"] - all_structures["homo"]

    assert all_structures.apply(
        lambda row: len(row.defect_representation) == len(defects.loc[row.descriptor_id, "defects"]), 
        axis=1).all()

    all_structures.to_pickle("datasets/all_structures_defects.pickle.gzip")


if __name__ == "__main__":
    main()


