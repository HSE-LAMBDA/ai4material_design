import os
import pandas as pd
import ase.io
import pymatgen.io.cif
from tqdm.auto import tqdm
from collections import defaultdict


def get_gpaw_trajectories(defect_db_path:str):
    res = defaultdict(list)
    for file_ in os.listdir(defect_db_path):
        if not file_.startswith("id"):
            continue
        this_folder = os.path.join(defect_db_path, file_, "relaxed", "trajectory")
        for traj_file in os.listdir(this_folder):
            try:
                res[file_].append(ase.io.read(os.path.join(this_folder, traj_file), index=":"))
            except ase.io.formats.UnknownFileTypeError:
                pass
    return res


def read_structures_descriptions(data_path:str):
    return pd.read_csv(os.path.join(data_path, "defects.csv"),
                       index_col="_id",
                       # An explicit list of columns is due to the fact that
                       # dichalcogenides8x8_innopolis_202108/defects.csv
                       # contains an unnamed index column, and
                       # datasets/dichalcogenides_innopolis_202105/defects.csv
                       # doesn't
                       usecols=["_id",
                                "descriptor_id",
                                "defect_id",
                                "energy",
                                "energy_per_atom",
                                "fermi_level",
                                "homo",
                                "lumo"])


def read_defects_descriptions(data_path:str):
    return pd.read_csv(
        os.path.join(data_path, "descriptors.csv"), index_col="_id",
        converters={"cell": eval, "defects": eval})    


def get_dichalcogenides_innopolis(data_path:str):
    structures = read_structures_descriptions(data_path)
    initial_structures = dict()
    structures_folder = os.path.join(data_path, "initial")
    for structure_file in tqdm(os.listdir(structures_folder)):
        this_file = pymatgen.io.cif.CifParser(os.path.join(structures_folder, structure_file))
        initial_structures[os.path.splitext(structure_file)[0]] = \
            this_file.get_structures(primitive=False)[0]
    structures["initial_structure"] = structures.apply(lambda row: initial_structures[row.name], axis=1)
    return structures, read_defects_descriptions(data_path)
