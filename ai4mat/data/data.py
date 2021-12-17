import os
import shutil
from pathlib import Path
import yaml
import pandas as pd
import ase.io
import pymatgen.io.cif
from tqdm.auto import tqdm
from collections import defaultdict

class StorageResolver:
    def __init__(self,
                 config_name=Path(__file__).parent.parent.parent.joinpath("storage.yaml")):
        self.root_folder = config_name.parents[0].absolute()
        with open(config_name) as config_file:
            self.config = yaml.safe_load(config_file)

    def __getitem__(self, key):
        return Path(self.root_folder, self.config[key])


NICE_TARGET_NAMES = {
    "homo": "HOMO, eV",
    "lumo": "LUMO, eV",
    "band_gap": "Band gap, eV",
    "formation_energy_per_site": "Formation energy per site, eV"
}


def get_experiment_name(experiment_path):
    return Path(experiment_path).name


def get_trial_name(trial_file):
    return Path(trial_file).stem


def get_prediction_path(experiment_name,
                        target_name,
                        this_trial_name):
    return Path(experiment_name,
                target_name,
                f"{this_trial_name}.csv.gz")


def get_targets_path(csv_cif_path):
    return Path(csv_cif_path.replace("csv_cif", "processed"), "targets.csv.gz")

def get_column_from_data_type(data_type):
    if data_type == 'sparse':
        return "defect_representation"
    elif data_type == 'full':
        return "initial_structure"
    else:
        raise ValueError("Unknown data_type")


def copy_indexed_structures(structures, input_folder, output_folder):
    save_path = Path(output_folder)
    save_path.mkdir(parents=True)
    # since we don't clean, raise if output exists
    for file_name in ("descriptors.csv", "elements.csv", "initial_structures.csv"):
        shutil.copy2(Path(input_folder, file_name),
                     save_path.joinpath(file_name))
    structures_folder = save_path.joinpath("initial")
    structures_folder.mkdir()
    input_structures_folder = Path(input_folder, "initial")
    for structure_id in structures.index:
        file_name = f"{structure_id}.cif"
        shutil.copy2(input_structures_folder.joinpath(file_name),
                     structures_folder.joinpath(file_name))
    structures.to_csv(save_path.joinpath("defects.csv"),
                      index_label="_id")

IS_INTENSIVE = {
    "homo": True,
    "LUMO": True,
    "formation_energy": False,
    "band_gap": True,
    "formation_energy_per_site": True
}


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
