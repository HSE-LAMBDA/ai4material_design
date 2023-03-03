import sys
import logging
import argparse
from pathlib import Path
from itertools import repeat
import shutil
from copy import deepcopy
import uuid
import pandas as pd
import yaml
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import DummySpecies, Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm.auto import trange

sys.path.append('.')
from ai4mat.data.data import (
    StorageResolver,
    Columns)


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


class MaxGenerationRetries(RuntimeError):
    """
    Reached the limit on the number of retries to generate a unique structure with defects.
    """
    pass


class StructureWriter:
    """
    Stores generated structures in the output directory.
    """
    def __init__(self,
                output_path: Path,
                clean_output: bool,
                base_material: str, # Used for id generation and defect description
                supercell: tuple[int, int, int]):
        self. descriptors_dict = {}
        self.COLUMNS = Columns()
        self.structures_dict = {
            self.COLUMNS["structure"]["id"]: [],
            self.COLUMNS["structure"]["descriptor_id"]: []
        }
        self.target_folder = Path(output_path)
        self.clean_output = clean_output
        self.structures_folder = self.target_folder / "poscars"
        self.base_material = base_material
        self.supercell = supercell

    def __enter__(self):
        if self.clean_output and self.target_folder.exists():
            shutil.rmtree(self.target_folder)
        self.target_folder.mkdir(parents=True, exist_ok=False)
        self.structures_folder.mkdir(exist_ok=False)
        return self

    def write(self, structure, description):
        compact_formula = structure.composition.formula.replace(" ", "")
        structure_id = "_".join((
            self.base_material,
            compact_formula,
            str(uuid.uuid4())))
        structure.to(filename=str(self.structures_folder / f"POSCAR_{structure_id}"), fmt="poscar")
        self.structures_dict["_id"].append(structure_id)
        if compact_formula not in self.descriptors_dict:
            self.descriptors_dict[compact_formula] = {
                "_id": str(uuid.uuid4()),
                "description": compact_formula,
                "base": self.base_material,
                "cell": self.supercell,
                "defects": description}
        self.structures_dict["descriptor_id"].append(self.descriptors_dict[compact_formula]["_id"])
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        descriptors_pd = pd.DataFrame.from_dict(list(self.descriptors_dict.values()))
        descriptors_pd.set_index(["_id"], inplace=True)
        descriptors_pd.to_csv(self.target_folder / "descriptors.csv")

        structures_pd = pd.DataFrame.from_dict(self.structures_dict)
        structures_pd.set_index([self.COLUMNS["structure"]["id"]], inplace=True)
        structures_pd.to_csv(self.target_folder / "defects.csv")


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

    with open(args.config_path, encoding="ascii") as config_file:
        config = yaml.safe_load(config_file)

    unit_cell = CifParser(StorageResolver()["materials"] /
        f"{config['base_material']}.cif").get_structures(primitive=False)
    if len(unit_cell) > 1:
        raise ValueError("Unit cell has multiple structures")
    unit_cell = unit_cell[0]

    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(config["supercell"])

    if not "all_combinations" in config or not config["all_combinations"]:
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
            print(max_defect_counts)
            raise ValueError(f"Can't reach the desired total defect count {absolute_max_defect_count} "
                            "under the per-type constraints.")
    
    with StructureWriter(args.output_path, args.clean_output, config["base_material"], config["supercell"]) as structure_writer:
        if "all_combinations" in config and config["all_combinations"]:
            # All defects from the list must be present exactly once
            # We do it naively and unsophisticatedly: fix the first defect, iterate over the rest
            one_defect_supercell = reference_supercell.copy()
            if len(config["defects"]) == 0:
                raise ValueError("No defects specified")
            description = []
            for element, replacement_list in config["defects"].items():
                for replacement in replacement_list:
                    if replacement == "Vacancy":
                        description.append({"type": "vacancy", "element": element})
                    else:
                        description.append({"type": "substitution", "from": element, "to": replacement})
            # Make the first defect
            # There is a catch. If we have a complicated cell (which we don't)
            # there might be more than two ways to place a single defect
            sga = SpacegroupAnalyzer(one_defect_supercell)
            symm_structure = sga.get_symmetrized_structure()
            if len(symm_structure.equivalent_sites) > 2:
                raise NotImplemented("Can't handle complicated cells (yet)")
            defects_to_place = config["defects"].copy()
            for i, site in enumerate(one_defect_supercell):
                if site.specie.symbol in defects_to_place:
                    replacement = defects_to_place[site.specie.symbol].pop()
                    if replacement == "Vacancy":
                        one_defect_supercell.remove_sites([i])
                    else:
                        one_defect_supercell.replace(i, species=Element(replacement))
                    if len(defects_to_place[site.specie.symbol]) == 0:
                        del defects_to_place[site.specie.symbol]
                    break
            if len(defects_to_place) == 1:
                element, replacement = next(iter(defects_to_place.items()))
                if len(replacement) > 1:
                    raise NotImplemented("Can't handle more than two defects yet")
                replacement = replacement[0]
                sga = SpacegroupAnalyzer(one_defect_supercell)
                symm_structure = sga.get_symmetrized_structure()
                for sites in symm_structure.equivalent_sites:
                    site = sites[0]
                    if site.specie.symbol == element:
                        two_defect_supercell = one_defect_supercell.copy()
                        site_index = two_defect_supercell.get_sites_in_sphere(site.coords, 0.1, include_index=True)[0][2]
                        if replacement == "Vacancy":
                            two_defect_supercell.remove_sites([site_index])
                        else:
                            two_defect_supercell.replace(site_index, species=Element(replacement))
                        structure_writer.write(two_defect_supercell.get_sorted_structure(), description)
            else:
                raise NotImplemented("Can't handle more than two defects yet")
        else:
            rng = np.random.default_rng(args.random_seed)
            for total_concentration in config["total_concentrations"]:
                try:
                    this_structures = []
                    target_total_defects = int(total_concentration*len(reference_supercell))
                    print(f"Generating structures with total concentration {total_concentration}; "
                        f"target defect count {target_total_defects}")
                    for _ in trange(config["structures_per_concentration"]):
                        is_unique = False
                        tries = 0
                        while not is_unique:
                            if tries > args.max_regeneration_retries:
                                raise MaxGenerationRetries(
                                    "Maximum number of tries exceeded for generating a unique structure.")
                            structure, description = generate_structure_with_random_defects(
                                target_total_defects, max_defect_counts, reference_supercell, rng, True)
                            tries += 1
                            is_unique = True
                            for previous_structure in this_structures:
                                if structure.matches(previous_structure):
                                    is_unique = False
                                    break
                        structure = structure.get_sorted_structure()
                        this_structures.append(structure)
                        structure_writer.write(structure, description)
                except MaxGenerationRetries:
                    if args.continue_on_max_regeneration_retries:
                        logging.warning("Maximum number of retries (%i) reached.",
                            args.max_regeneration_retries)
                        logging.warning("Concentration %f has %i structures instead of %i requested.",
                                        total_concentration, len(this_structures),
                                        config['structures_per_concentration'])
                        continue
                    else:
                        raise


if __name__ == "__main__":
    main()
