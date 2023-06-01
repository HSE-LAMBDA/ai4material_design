import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.sites import PeriodicSite


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
    structure = structure.copy()

    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)
    # reference_sites = get_frac_coords_set(reference_supercell)

    defects = []
    full_were_species = []
    defect_energy_correction = 0

    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)

    for coords, reference_site in reference_structure_dict.items():
        # Vacancy
        reference_site.properties['was'] = reference_site.specie.Z

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