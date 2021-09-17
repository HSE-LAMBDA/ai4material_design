import dill as pickle
import os
import numpy as np
import json

import pymatgen
from pymatgen.analysis.defects.core import Vacancy, Substitution
from pymatgen.core import Element, Site
from pymatgen.io.cif import CifParser

BASE_PATH = './molecules'

def is_copy(structure: pymatgen.core.Structure,
            generated_structures: list) -> bool:
    """
    Determines if there is an invariant found defect for the given one
    """
    is_copy = False
    for another_structure in generated_structures:
        if structure.matches(another_structure):
            is_copy = True
            break
    return is_copy


def check_in(site: pymatgen.core.sites.PeriodicSite, defect_sites: list):
    """
    Determines if the given site is in the list of sites
    """
    is_in = False
    for defect_site in defect_sites:
        if (site.coords == defect_site.coords).all():
            is_in = True
    return is_in


def generate_one_random_defect(defect_descriptor: dict, structure: pymatgen.core.Structure):
    """
    Generates a random defect by the rules, described in **defect_descriptor**
    **structure** is the base for defect generation, e.g. MoS2 6x6 cell without defected sites
    """
    number_of_changes = len(defect_descriptor['defects'])
    defected_sites = []
    for i in range(number_of_changes):
        site = np.random.choice(defect_descriptor['defects'][i]['sites'])                       
        while check_in(site, defected_sites):
            site = np.random.choice(defect_descriptor['defects'][i]['sites'])
        defected_sites.append(site) 
        if (defect_descriptor['defects'][i]['type'] == 'substitution'):
            gen = Substitution(structure, defect_site=site)
        elif (defect_descriptor['defects'][i]['type'] == 'vacancy'):
            gen = Vacancy(structure, defect_site=site)
        structure = gen.generate_defect_structure()
    return structure


def prepare_sites(defect_descriptor: dict, 
                  bulk_structure: pymatgen.core.structure.IStructure
                 ) -> None:
    """ 
    Computes all possible sites for each defect in
    defect_descriptor["defects"] (list of applied defects)
    
    Saves the sites into defect_descriptor
    """
    for defect in defect_descriptor['defects']:
        if (defect["type"] == 'substitution'):
            defect["sites"] = []
            for site in bulk_structure.copy():
                if (site.specie == Element(defect['from'])):
                    site.species = Element(defect["to"])
                    defect['sites'].append(site)
        if (defect["type"] == 'vacancy'):
            defect["sites"] = []
            for site in bulk_structure.copy():
                if (site.specie == Element(defect['element'])):
                    defect['sites'].append(site)


def create_supercell(defect_descriptor: dict) -> pymatgen.core.structure.Structure:
    """ Applies supercell to the defect """
    bulk_structure = CifParser(os.path.join(BASE_PATH, f'''{defect_descriptor['base']}.cif''')).get_structures()[0]
    bulk_structure.make_supercell(defect_descriptor["cell"])
    return bulk_structure


def generate_defect_set(defect_descriptor: dict) -> None:
    """
    Computes set of defects
    Args:
        defect_descriptor (dict): Dictionary with fields "cell", "defects_number", "defects", "base"
        Descriptor examples can ve found in folder screening_pipeline/descriptors
        
        database (str): Where to add the defect
    """
    bulk_structure = create_supercell(defect_descriptor)
    prepare_sites(defect_descriptor, bulk_structure.copy())
    generated_set = []
    # counter counts the number of trials for defect creation
    counter = 0
    while True:
        counter += 1
        structure = generate_one_random_defect(defect_descriptor, bulk_structure.copy())
        if not is_copy(structure, generated_set):
            generated_set.append(structure)
        if len(generated_set) / counter < 0.01:
            # if the number of defect creation trials is 100 times more than the number
            # of generated defects, than we probably generated all possible defects
            # my experience shows that 100 in a reasonable coefficient
            return generated_set
