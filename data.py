

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


def get_dichalcogenides_innopolis(data_path:str):
    structures = pd.read_csv(os.path.join(data_path, "defects.csv"))
    initial_structures = dict()
    structures_folder = os.path.join(data_path, "initial")
    for structure_file in tqdm(os.listdir(structures_folder)):
        this_file = pymatgen.io.cif.CifParser(os.path.join(structures_folder, structure_file))
        initial_structures[os.path.splitext(structure_file)[0]] = this_file.get_structures(primitive=False)[0]
    structures["initial_structure"] = structures.apply(lambda row: initial_structures[row._id], axis=1)
    return structures
