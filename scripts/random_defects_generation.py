from collections import defaultdict
from email.policy import default
from itertools import repeat
from copy import deepcopy
import uuid
import pandas as pd
import sys
import logging
import argparse
import yaml
from pathlib import Path
from pymatgen.io.cif import CifParser
import numpy as np
from pymatgen.core.periodic_table import DummySpecies, Element
from pymatgen.core.structure import Structure
import shutil
from tqdm.auto import trange

sys.path.append('.')
from ai4mat.data.data import (
    StorageResolver,
    Columns)


def generate_structure_with_defects(
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

class MaxGenerationRetries(RuntimeError): pass

def main():
    parser = argparse.ArgumentParser(description='Generate structures with random defects')
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to the generation config file")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument("--random-seed", type=int, help="Random seed")
    parser.add_argument("--max-regeneration-retries", type=int, default=10,
                        help="Maximum number of retries to generate a unique structure")
    parser.add_argument("--continue-on-max-regeneration-retries", action="store_true",
                        help="Contiue generating to the next concentrations if "
                        "the maximum number of retries is reached when trying "
                        "to generate unique structures.")
    parser.add_argument("--clean-output", action="store_true",
                        help="Clean the output directory before generating")
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    unit_cell = CifParser(StorageResolver()["materials"] /
        f"{config['base_material']}.cif").get_structures(primitive=False)
    if len(unit_cell) > 1:
        raise ValueError("Unit cell has multiple structures")
    unit_cell = unit_cell[0]
    
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(config["supercell"])

    rng = np.random.default_rng(args.random_seed)

    element_counts = reference_supercell.composition.as_dict()
    max_defect_counts = {}
    max_allowed_defects = 0
    for element, replacement_list in config["defects"].items():
        max_for_element = int(config["max_concentration_per_type"]*element_counts[element])
        if max_for_element > 0:
            max_allowed_defects += max_for_element*len(replacement_list)
            max_defect_counts[element] = dict(zip(
                replacement_list,
                repeat(max_for_element)))
    
    absolute_max_defect_count = int(max(config["total_concentrations"])*len(reference_supercell))
    if max_allowed_defects < absolute_max_defect_count:
        raise ValueError(f"Can't reach the desired total defect count {absolute_max_defect_count} "
                         "under the per-type constraints.")

    descriptors_dict = {}
    COLUMNS = Columns()
    structures_dict = {
        COLUMNS["structure"]["id"]: [],
        COLUMNS["structure"]["descriptor_id"]: []
    }
    target_folder = Path(args.output_path)

    if args.clean_output and target_folder.exists():
        shutil.rmtree(target_folder)

    target_folder.mkdir(parents=True, exist_ok=False)
    structures_folder = target_folder / "poscars"
    structures_folder.mkdir(exist_ok=False)

    for total_concentration in config["total_concentrations"]:
        try:
            this_structures = []
            print(f"Generating structures with total concentration {total_concentration}")
            for _ in trange(config["structures_per_concentration"]):
                target_total_defects = int(total_concentration*len(reference_supercell))

                is_unique = False
                tries = 0
                while not is_unique:
                    if tries > args.max_regeneration_retries:
                        raise MaxGenerationRetries(
                            "Maximum number of tries exceeded for generating a unique structure.")
                    structure, description = generate_structure_with_defects(
                        target_total_defects, max_defect_counts, reference_supercell, rng, True)
                    tries += 1
                    is_unique = True
                    for previous_structure in this_structures:
                        if structure.matches(previous_structure):
                            is_unique = False
                            break
                structure = structure.get_sorted_structure()
                this_structures.append(structure)
                compact_formula = structure.composition.formula.replace(" ", "")
                structure_id = "_".join((
                    config['base_material'],
                    compact_formula,
                    str(uuid.uuid4())))
                structure.to("POSCAR", structures_folder / f"POSCAR_{structure_id}")
                structures_dict["_id"].append(structure_id)
                if compact_formula not in descriptors_dict:
                    descriptors_dict[compact_formula] = {
                        "_id": str(uuid.uuid4()),
                        "description": compact_formula,
                        "base": config["base_material"],
                        "cell": config["supercell"],
                        "defects": description}
                structures_dict["descriptor_id"].append(descriptors_dict[compact_formula]["_id"])
        except MaxGenerationRetries:
            if args.continue_on_max_regeneration_retries:
                logging.warning(f"Maximum number of retries ({args.max_regeneration_retries}) reached.")
                logging.warning(f"Concentration {total_concentration} has {len(this_structures)} structures "
                                f"instead of {config['structures_per_concentration']} requested.")
                continue
            else:
                raise
    
    descriptors_pd = pd.DataFrame.from_dict(list(descriptors_dict.values()))
    descriptors_pd.set_index(["_id"], inplace=True)
    descriptors_pd.to_csv(target_folder / "descriptors.csv")

    structures_pd = pd.DataFrame.from_dict(structures_dict)
    structures_pd.set_index([COLUMNS["structure"]["id"]], inplace=True)
    structures_pd.to_csv(target_folder / "defects.csv")

if __name__ == "__main__":
    main()