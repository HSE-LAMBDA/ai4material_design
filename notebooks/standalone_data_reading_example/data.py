from typing import Union
import os
from pathlib import Path
import yaml
import tarfile
from tqdm import tqdm
import pandas as pd
import pymatgen.io.cif


class Columns(dict):
    def __init__(self,
                 config_name=Path(__file__).parent.joinpath("data_format.yaml")):
        with open(config_name) as config_file:
            config = yaml.safe_load(config_file)
        super().__init__(config)

def read_structures_descriptions(data_path:Path) -> pd.DataFrame:
    """
    Reads the description of the structures in the folder.
    We assume that all columns not in Column enum are targets.
    Args:
        data_path: path to the folder with the data
    Returns:
        pandas DataFrame with the description of the structures
    """
    try:
        return pd.read_csv(data_path / "defects.csv.gz",
                           index_col=Columns()["structure"]["id"])
    except FileNotFoundError:
        logging.warn(f"Deprecation warning: defects.csv {data_path} should be defects.csv.gz")
        return pd.read_csv(data_path / "defects.csv",
                           index_col=Columns()["structure"]["id"])


def read_defects_descriptions(data_path:Path):
    return pd.read_csv(
        data_path / "descriptors.csv", index_col=Columns()["defect"]["id"],
        converters={"cell": lambda x: tuple(eval(x)), "defects": eval})


def read_csv_cif(data_path: Union[str,Path]):
    data_path = Path(data_path)
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