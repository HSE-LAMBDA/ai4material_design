import pandas as pd
import sys
import os
from typing import List
from rolos_sdk import Dataframe, DataStorageInterface, DataStorageType, TableColumn
from rolos_sdk.structures.object.pymatgen import PyMatGenObject
from multiprocessing import Pool, get_context
from itertools import chain, repeat
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from ai4mat.data.data import StorageResolver, read_defects_descriptions

table_structure = {
    "initial_structure": PyMatGenObject,
    "defect_representation": PyMatGenObject,
    "formation_energy_per_site": float,
    "homo_lumo_gap_min": float,
    "description": str,
    "base": str,
    "cell": str,
    "defects": str,
    'energy': float,
    'formation_energy': float,
    'total_mag': float,
    'fermi_level':float,
    'homo': float,
    'lumo': float,
    'homo_lumo_gap_majority': float,
    'lumo_majority': float,
    'homo_majority': float,
    'homo_lumo_gap_minority': float,
    'lumo_minority': float,
    'homo_minority': float,
    'homo_lumo_gap_max': float
}

def upload_data(table_data: pd.DataFrame, table_schema: List[TableColumn], name: str) -> None:
    with DataStorageInterface.create(DataStorageType.Datacat) as storage:
        with Dataframe(name=name, storage=storage, schema=table_schema) as frame:
            frame.insert(table_data.values.tolist())


def prepare_dataset(dataset):
    data = pd.read_pickle(StorageResolver()["processed"] / dataset / "data.pickle.gz")
    data_description = read_defects_descriptions(StorageResolver()["csv_cif"] / dataset)
    data = data.join(data_description, on="descriptor_id", how="left")
    for column_of_opportunity in ("homo", "lumo"):
        if column_of_opportunity not in data.columns:
            data[column_of_opportunity] = None
    for field, rolos_type in table_structure.items():
        if rolos_type is str:
            data[field] = data[field].astype(str)
    return data[table_structure.keys()]


def main():
    datasets = {
            "high_density_defects/MoS2_500": "MoS2 high concentration",
            "high_density_defects/WSe2_500": "WSe2 high concentration",
            "low_density_defects/MoS2": "MoS2 low concentration",
            "low_density_defects/WSe2": "WSe2 low concentration",
            "high_density_defects/BP_spin_500": "BP high concentration",
            "high_density_defects/hBN_spin_500": "hBN high concentration",
            "high_density_defects/GaSe_spin_500": "GaSe high concentration",
            "high_density_defects/InSe_spin_500": "InSe high concentration"
    }
    available_CPUs = int(os.environ["ROLOS_AVAILABLE_CPU"])
    multiprocess_method = "spawn"
    print("Reading datasets")
    with get_context(multiprocess_method).Pool(min(len(datasets), available_CPUs)) as pool:
        combined_data = pool.map(prepare_dataset, datasets)
    # print("Combining datasets")
    # combined_data_pd = pd.concat(combined_data, axis=0)
    print("Uploading datasets")
    schema = [TableColumn(name=key, type=value) for key, value in table_structure.items()]
    # upload_data(combined_data_pd, schema, "Combined 2DMD dataset")
    for dataset_pd, name in zip(combined_data, datasets.values()):
        upload_data(dataset_pd, schema, name)
    # with get_context(multiprocess_method).Pool(min(len(datasets) + 1, available_CPUs)) as pool:
    #    pool.starmap(upload_data, chain(
    #        zip(combined_data, repeat(schema), datasets.values()),
    #        ((combined_data_pd, schema, "Combined 2DMD dataset"),)))


if __name__ == "__main__":
    main()