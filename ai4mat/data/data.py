from enum import Enum
import logging
import os
import shutil
from pathlib import Path
import yaml
import pandas as pd
import ase.io
import pymatgen.io.cif
from tqdm.auto import tqdm
from collections import defaultdict
import tarfile


TRAIN_FOLD = 0
TEST_FOLD = 1
VALIDATION_FOLD = 2

NICE_TARGET_NAMES = {
    "homo": "HOMO, eV",
    "lumo": "LUMO, eV",
    "band_gap": "Band gap, eV",
    "formation_energy_per_site": "Formation energy per site, eV"
}

class Columns(dict):
    def __init__(self,
                 config_name=Path(__file__).parent.parent.parent.joinpath("data_format.yaml")):
        with open(config_name) as config_file:
            config = yaml.safe_load(config_file)
        super().__init__(config)


class StorageResolver:
    def __init__(self,
                 config_name=Path(__file__).parent.parent.parent.joinpath("storage.yaml")):
        self.root_folder = config_name.parents[0].absolute()
        with open(config_name) as config_file:
            self.config = yaml.safe_load(config_file)

    def __getitem__(self, key):
        return Path(self.root_folder, self.config[key])


class Is_Intensive:
    def __init__(self):
        self.attr = {
            "energy": False,
            "homo": True,
            "normalized_homo": True,
            "homo_1": True,
            "homo_2": True,
            "lumo": True,
            "normalized_lumo": True,
            "lumo_1": True,
            "lumo_2": True,
            "formation_energy": False,
            "energy_per_atom": True,
            "band_gap": True,
            "band_gap_1": True,
            "band_gap_2": True,
            "band_gap_majority": True,
            "band_gap_minority": True,
            "homo_lumo_gap": True,
            "homo_lumo_majority": True,
            "homo_lumo_minority": True,
            "homo_lumo_gap_min": True,
            "homo_lumo_gap_max": True,
            "homo_majority": True,
            "homo_minority": True,
            "homo_min": True,
            "homo_max": True,
            "lumo_majority": True,
            "lumo_minority": True,
            "lumo_min": True,
            "lumo_max": True,
            "normalized_homo_majority": True,
            "normalized_homo_minority": True,
            "normalized_homo_max": True,
            "normalized_homo_min": True,
            "normalized_lumo_majority": True,
            "normalized_lumo_minority": True,
            "normalized_lumo_max": True,
            "normalized_lumo_min": True,
            "formation_energy_per_site": True,
            "band_gap_from_eigenvalue_band_properties": True,
            "band_gap_from_get_band_structure": True,
            "total_mag": True
        }

    def __getitem__(self, item):
        if isinstance(item, list):
            return True
        return self.attr[item]


class DataLoader:
    def __init__(self, dataset_paths, folds_index):
        self.dataset_paths = dataset_paths
        self.datasets = dict()
        self.targets = None
        self.folds_index = folds_index

    def _load_data(self, filename):
        if filename.endswith(".pickle.gz"):
            read_func = pd.read_pickle
        elif filename.endswith(".csv.gz") or filename.endswith(".csv"):
            read_func = pd.read_csv
        else:
            raise ValueError("Unknown file type")

        storage_resolver = StorageResolver()
        data = pd.concat(
            [
                read_func(
                    storage_resolver["processed"].joinpath(
                        path, filename
                    )
                )
                for path in self.dataset_paths
            ],
            axis=0,
        )
        return data

    def _load_matminer(self):
        return self._load_data("matminer.csv.gz").set_index("_id").reindex(self.folds_index)

    def _load_sparse(self):
        data = self._load_data("data.pickle.gz")
        return data[get_column_from_data_type("sparse")].reindex(self.folds_index)

            
    
    def _load_full(self):
        return self._load_data("data.pickle.gz")[get_column_from_data_type("full")].reindex(self.folds_index)

    def _load_targets(self):
        return self._load_data("targets.csv.gz").set_index("_id").reindex(self.folds_index)

    def get_structures(self, representation):
        """
        Lazyly loads corresponding representation and returns it as a pandas Series or DataFrame
        """
        if representation == "full":
            if "full" not in self.datasets:
                self.datasets["full"] = self._load_full()
        elif representation == "sparse":
            if "sparse" not in self.datasets:
                self.datasets["sparse"] = self._load_sparse()
        elif representation == "matminer":
            if "matminer" not in self.datasets:
                self.datasets["matminer"] = self._load_matminer()
        else:
            raise ValueError("Unknown data representation requested")
        return self.datasets[representation]

    def get_targets(self, target_name):
        if self.targets is None:
            self.targets = self._load_targets()
        return self.targets[target_name]


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


def get_matminer_path(csv_cif_path):
    return Path(csv_cif_path.replace("csv_cif", "processed"), "matminer.csv.gz")


def get_column_from_data_type(data_type):
    if data_type == 'sparse':
        return "defect_representation"
    elif data_type == 'full':
        return "initial_structure"
    elif data_type == 'matminer':
        return "matminer"
    else:
        raise ValueError("Unknown data_type")


def copy_indexed_structures(
    index: pd.Index,
    input_tar: Path,
    output_tar: Path) -> None:
    copied = pd.Series(data=False, index=index, dtype=bool)
    with tarfile.open(input_tar, "r:gz") as input_tar_file, \
        tarfile.open(output_tar, "w:gz") as output_tar_file:
        for member in tqdm(input_tar_file.getmembers()):
            assert member.name.endswith(".cif")
            structure_id = member.name[:-4]
            if structure_id in index:
                output_tar_file.addfile(member, input_tar_file.extractfile(member))
                copied[structure_id] = True
    if not copied.all():
        raise ValueError("Not all structures were copied")


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


def read_structures_descriptions(data_path) -> pd.DataFrame:
    """
    Reads the description of the structures in the folder.
    We assume that all columns not in Column enum are targets.
    Args:
        data_path: path to the folder with the data
    Returns:
        pandas DataFrame with the description of the structures
    """
    try:
        return pd.read_csv(os.path.join(data_path, "defects.csv.gz"),
                           index_col=Columns()["structure"]["id"])
    except FileNotFoundError:
        logging.warn(f"Deprecation warning: defects.csv {data_path} should be defects.csv.gz")
        return pd.read_csv(os.path.join(data_path, "defects.csv"),
                           index_col=Columns()["structure"]["id"])


def read_defects_descriptions(data_path:str):
    return pd.read_csv(
        os.path.join(data_path, "descriptors.csv"), index_col="_id",
        converters={"cell": lambda x: tuple(eval(x)), "defects": eval})


def get_dichalcogenides_innopolis(data_path):
    structures = read_structures_descriptions(data_path)
    initial_structures = dict()
    structures_tar = Path(data_path) / "initial.tar.gz"
    try:
        with tarfile.open(structures_tar, "r:gz") as tar:
            for member in tqdm(tar.getmembers()):
                assert member.name.endswith(".cif")
                structure_id = os.path.splitext(member.name)[0]
                this_structure_file = pymatgen.io.cif.CifParser.from_string(tar.extractfile(member).read().decode("ascii"))
                initial_structures[structure_id] = this_structure_file.get_structures(primitive=False)[0]
    except FileNotFoundError as e:
        logging.warning(e)
        logging.warning('Trying obsolete format (folder without .tar.gz)')
        structures_folder = os.path.join(data_path, "initial")
        for structure_file in tqdm(os.listdir(structures_folder)):
            this_file = pymatgen.io.cif.CifParser(os.path.join(structures_folder, structure_file))
            initial_structures[os.path.splitext(structure_file)[0]] = \
            this_file.get_structures(primitive=False)[0]
        logging.warn(f"Data in {data_path} is in obsolete format")
    structures[Columns()["structure"]["unrelaxed"]] =  structures.apply(
        lambda row: initial_structures[row.name], axis=1)
    return structures, read_defects_descriptions(data_path)


def read_experiment_datasets(experiment_name):
    storage_resolver = StorageResolver()
    experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    folds = pd.read_csv(storage_resolver["experiments"].joinpath(
                        experiment_name).joinpath("folds.csv.gz"),
                        index_col="_id")
    datasets = pd.concat([pd.read_pickle(storage_resolver["processed"]/dataset/"data.pickle.gz") for dataset in experiment["datasets"]], axis=0)
    defects = pd.concat([read_defects_descriptions(storage_resolver["csv_cif"]/dataset) for dataset in experiment["datasets"]], axis=0)
    return experiment, folds, datasets, defects