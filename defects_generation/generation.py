import os
import json

import pymatgen
from pymatgen.analysis.defects.core import Vacancy, Substitution
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import PointDefectComparator
from pymatgen.core import Element, Site
from pymatgen.io.cif import CifParser

BASE_PATH = './molecules'

def is_copy(current_generator: pymatgen.analysis.defects.core.Defect,
            generated_set: list) -> bool:
    """
    Determines if there is an invariant found defect for the given one
    """
    is_copy = False
    for generator in generated_set:
        comparator = PointDefectComparator()
        if (comparator.are_equal(current_generator, generator)):
            is_copy = True
            break
    return is_copy


def generate(defect_descriptor: dict,
             structure: pymatgen.core.structure.IStructure,
             level: int =0,
             defected_coords: list =[],
             generated_set: list = [],
            ) -> None:
    """ 
    Recursive search for the defect 
    Args: 
        defect_descriptor : Dictionary with fields "cell", "defects_number", "defects", "base"
        Descriptor examples can ve found in folder screening_pipeline/descriptors
        
        structure: base structure for single defect
        
        level: recursion depth = number of generated defects in **strucure**
        
        defected_coords: list of coordinates with generated defects, len(defected_coords) == level
        
        generated_set: list of generated defects, it is used to define whether the current defect
        is invariant to one of the computed
    """    
    defect = defect_descriptor['defects'][level]
    
    if (level == 0):
        if (defect['type'] == 'substitution'):
            gen = Substitution(structure, defect_site=defect['sites'][0])
        elif (defect['type'] == 'vacancy'):
            gen = Vacancy(structure, defect_site=defect['sites'][0])
        
        defect_structure = gen.generate_defect_structure()
        generate(defect_descriptor, defect_structure.copy(), 1,
                             [defect['sites'][0].coords], generated_set)  
        
    elif (level == len(defect_descriptor['defects']) - 1):
        for i, site in enumerate(defect['sites']):
            # check if the given site is already defected
            to_cont = False
            for defected_coord in defected_coords:
                if (site.coords == defected_coord).all():
                    to_cont = True
                    break
            if to_cont: 
                continue
                    
            # point defect
            if (defect['type'] == 'substitution'):
                gen = Substitution(structure, defect_site=site)     
            if (defect['type'] == 'vacancy'):
                gen = Vacancy(structure, defect_site=site)
                
            if not is_copy(gen, generated_set):
                generated_set.append(gen)
    else:
        for i, site in enumerate(defect['sites']):
            # check if the given site is already defected
            to_cont = False
            for defected_coord in defected_coords:
                if (site.coords == defected_coord).all():
                    to_cont = True
                    break
            if to_cont: 
                continue
                
            # point defect
            if (defect['type'] == 'substitution'):
                gen = Substitution(structure, defect_site=site)
            if (defect['type'] == 'vacancy'):
                gen = Vacancy(structure, defect_site=site)
            defect_structure = gen.generate_defect_structure()
            next_defect_coords = defected_coords.copy()
            next_defect_coords.append(site.coords)
            generate(defect_descriptor, defect_structure.copy(), level+1,
                                 next_defect_coords, generated_set)

            
def prepare_sites(defect_descriptor: dict, 
                  bulk_structure: pymatgen.core.structure.Structure
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
    bulk_structure = CifParser(
        os.path.join(BASE_PATH, f'''{defect_descriptor['base']}.cif''')
    ).get_structures()[0]
    bulk_structure.make_supercell(defect_descriptor["cell"])
    return bulk_structure


def generate_defect_set(defect_descriptor: dict) -> None:
    """
    Computes set of defects
    Args:
        defect_descriptor (dict): Dictionary with fields "cell", "defects_number", "defects", "base"
        Descriptor examples are attached
    """
    
    bulk_structure = create_supercell(defect_descriptor)
    prepare_sites(defect_descriptor, bulk_structure.copy())
    
    generated_set = []
    generate(defect_descriptor, bulk_structure.copy(), generated_set = generated_set)
    
    for i in range(len(generated_set)):
        generated_set[i] = generated_set[i].generate_defect_structure()
    return generated_set