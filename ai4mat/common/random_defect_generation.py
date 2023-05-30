from copy import deepcopy
import numpy as np
from pymatgen.core import Structure, Element, DummySpecies

def generate_structure_with_random_defects(
    target_total_defects: int,
    max_defect_counts: dict[dict[int]],
    reference_supercell: Structure,
    rng: np.random.Generator,
    generate_innopolis_descritption: bool) -> Structure:
    """"
    Generates a structure with defects.

    We have the following constraints for the defect distribution:
    1. The total number of defects is fixed
    2. The total number of defects of each type must not exceed the defined fraction
       of the available sites of the corresponding element
    3. Defects within the same pristine element type are equally probable
    4. Defect probability for each site is the same
    Args:
        target_total_defects: number of defects in the structure
        max_defect_counts: maximum number of defects for each element and defect type
        reference_supercell: reference supercell
        rng: random number generator
    Returns:
        Structure with defects
    """
    supercell = reference_supercell.copy()
    if target_total_defects == 0:
        if generate_innopolis_descritption:
            return supercell, []
        else:
            return supercell
    remaining_defect_counts = deepcopy(max_defect_counts)
    total_defects = 0
    permutation = rng.permutation(len(supercell))
    if generate_innopolis_descritption:
        defects_list = []
    for i in permutation:
        if supercell[i].specie.symbol not in remaining_defect_counts:
            continue
        replacement = rng.choice(tuple(remaining_defect_counts[supercell[i].specie.symbol].keys()))
        if generate_innopolis_descritption:
            if replacement == "Vacancy":
                defects_list.append({"type": "vacancy", "element": supercell[i].specie.symbol})
            else:
                defects_list.append({
                    "type": "substitution",
                    "from": supercell[i].specie.symbol,
                    "to": replacement})
        if remaining_defect_counts[supercell[i].specie.symbol][replacement] <= 1:
            del remaining_defect_counts[supercell[i].specie.symbol][replacement]
            if len(remaining_defect_counts[supercell[i].specie.symbol]) == 0:
                del remaining_defect_counts[supercell[i].specie.symbol]
        else:
            remaining_defect_counts[supercell[i].specie.symbol][replacement] -= 1
        total_defects += 1
        supercell.replace(i,
            species=DummySpecies() if replacement == "Vacancy" else Element(replacement))
        if total_defects == target_total_defects:
            break
    supercell.remove_sites([i for i, site in enumerate(supercell) if site.specie == DummySpecies()])
    if generate_innopolis_descritption:
        return supercell, defects_list
    else:
        return supercell
