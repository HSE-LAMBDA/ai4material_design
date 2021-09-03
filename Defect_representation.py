#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from data import get_dichalcogenides_innopolis


# In[2]:


from pymatgen.core.sites import PeriodicSite
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies, Element
from pymatgen.io.cif import CifParser


# In[3]:


structures = get_dichalcogenides_innopolis("datasets/dichalcogenides_innopolis_202105/")


# In[4]:


structures_8x8 = get_dichalcogenides_innopolis("datasets/dichalcogenides8x8_innopolis_202108/").dropna().reset_index(drop=True)


# In[5]:


# TODO(inner perfectionist) eval is unsecure
defects_2015 = pd.read_csv(
  "datasets/dichalcogenides_innopolis_202105/descriptors.csv", index_col="_id",
  converters={"cell": eval, "defects": eval})
defects_8x8 = pd.read_csv(
  "datasets/dichalcogenides8x8_innopolis_202108/descriptors.csv", index_col="_id",
  converters={"cell": eval, "defects": eval})
defects = pd.concat([defects_2015, defects_8x8], axis=0)


# In[6]:


materials = defects.base.unique()


# In[7]:


unit_cells = {}
for material in materials:
  unit_cells[material] = CifParser(os.path.join(
  "defects_generation", "molecules", f"{material}.cif")).get_structures(primitive=True)[0]


# In[8]:


initial_structure_properties = pd.concat([
  pd.read_csv(os.path.join("datasets", "dichalcogenides_innopolis_202105", "initial_structures.csv"),
              index_col=["base", "cell_length"], usecols=[1,2,3,4]),
  pd.read_csv(os.path.join("datasets", "dichalcogenides8x8_innopolis_202108", "initial_structures.csv"),
              index_col=["base", "cell_length"], usecols=[1,2,3,4])
], axis=0)


# In[9]:


single_atom_energies = pd.read_csv(os.path.join("datasets", "single_atom_energies.csv"),
                                   index_col=0,
                                   converters={0: Element})
SINGLE_ENENRGY_COLUMN = "energy_bulk_estimate"
temporary_energies = {"Mo": -10.85, "W": -12.96, "S": -4.24, "Se": -3.50, "O": -4.95}
for element_name, energy in temporary_energies.items():
  single_atom_energies.loc[Element(element_name), SINGLE_ENENRGY_COLUMN] = energy


# In[10]:


def get_frac_coords_set(structure):
  return set(map(tuple, np.round(structure.frac_coords, 3)))


# In[11]:


def strucure_to_dict(structure, precision=3):
  res = {}
  for site in structure:
    res[tuple(np.round(site.frac_coords, precision))] = site
  return res


# In[12]:


def get_defects(structure, defect_description):
  unit_cell = unit_cells[defect_description.base]
  reference_species = set(unit_cell.species)
  reference_supercell = unit_cell.copy()
  reference_supercell.make_supercell(defect_description.cell)
  reference_sites = get_frac_coords_set(reference_supercell)

  defects = []
  were_species = []
  initial_energy = initial_structure_properties.loc[defect_description.base, defect_description.cell[0]].energy
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
      initial_energy += single_atom_energies.loc[
        structure_dict[coords].specie, SINGLE_ENENRGY_COLUMN]
      defect_energy_correction += single_atom_energies.loc[
        reference_site.specie, SINGLE_ENENRGY_COLUMN]
    
  res = Structure(lattice=structure.lattice,
                  species=[x.specie for x in defects],
                  coords=[x.frac_coords for x in defects],
                  site_properties={"was": were_species},
                  coords_are_cartesian=False)
  res.state = [sorted([element.Z for element in reference_species])]
  return res, defect_energy_correction - initial_energy


# In[13]:


# TODO(kazeevn) this all is very ugly
def get_defecs_from_row(row):
  defect_structure, formation_energy_part = get_defects(row.initial_structure, defects.loc[row.descriptor_id])
  return defect_structure, formation_energy_part + row.energy


# In[14]:


all_stuctures = pd.concat([structures, structures_8x8], axis=0, ignore_index=True)
defect_properties = all_stuctures.apply(
  get_defecs_from_row,
  axis=1,
  result_type="expand")
defect_properties.columns = ["defect_representation", "formation_energy"]
all_stuctures = all_stuctures.join(defect_properties)


# In[15]:


# Test
assert all_stuctures.apply(
  lambda row: len(row.defect_representation) == len(defects.loc[row.descriptor_id, "defects"]), 
  axis=1).all()


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


all_stuctures.to_pickle("datasets/all_structures_defects.pickle.gzip")

def is_vacancy_only(defect_structure):
  return all((isinstance(specie, DummySpecies) for specie in defect_structure.species))

def is_clean(row):
  """
  formation energy > -4 eV/defect site for structured with at least 1 substitution
  formation energy > 0 for vacancy-only
  """
  if is_vacancy_only(row.defect_representation):
    return row.formation_energy > 0
  else:
    return row.formation_energy > -4*len(row.defect_representation)

clean_structures = all_stuctures[all_stuctures.apply(is_clean, axis=1)]
train, test = train_test_split(clean_structures, test_size=0.25, random_state=2141)
train.to_pickle("datasets/train_defects.pickle.gzip")
test.to_pickle("datasets/test_defects.pickle.gzip")


# In[18]:


vac_only = clean_structures.defect_representation.apply(is_vacancy_only)
vac_train, vac_test = train_test_split(clean_structures[vac_only], test_size=0.25, random_state=211231)
vac_train.to_pickle("datasets/train_defects_vac_only.pickle.gzip")
vac_test.to_pickle("datasets/test_defects_vac_only.pickle.gzip")


# In[19]:


is_small = clean_structures.apply(
  lambda row: defects.loc[row.descriptor_id, "cell"][0] < 8,
  axis=1)
vac_8x8 = clean_structures[vac_only & ~is_small]
vac_no_8x8 = clean_structures[vac_only & is_small]
vac_no_8x8.to_pickle("datasets/train_defects_vac_only_no_8x8_in_train.pickle.gzip")
vac_8x8.to_pickle("datasets/test_defects_vac_only_no_8x8_in_train.pickle.gzip")


# In[20]:


train_8x8, test_8x8 = train_test_split(vac_8x8, test_size=0.5, random_state=42134114)
pd.concat([vac_no_8x8, train_8x8], ignore_index=True).to_pickle(
  "datasets/train_defects_vac_only_8x8_split.pickle.gzip")
test_8x8.to_pickle(
  "datasets/test_defects_vac_only_8x8_split.pickle.gzip")


# In[21]:


from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor


# In[22]:


structure_to_plot = all_stuctures.iloc[1221]
view(AseAtomsAdaptor().get_atoms(structure_to_plot.initial_structure), viewer='ngl')


# In[23]:


view(AseAtomsAdaptor().get_atoms(structure_to_plot.defect_representation), viewer='ngl')

