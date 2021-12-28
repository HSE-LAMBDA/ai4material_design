import tqdm
import torch

import numpy as np
import pandas as pd

from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from ai4mat.common.atom_to_graph import AtomsToGraphs
from ai4mat.common.utils import cache
from functools import lru_cache
import pdb

class GemNetFullStructFolds:
    def __init__(self, 
            train_structures,
            train_targets,
            test_structures,
            test_targets,
            configs,
            graph_construction_config=None,
            ):
    
        self.graph_construction_config = graph_construction_config
        self.config = configs
        
        self.train_graph_list = self.construct_dataset(train_structures, train_targets)
        self.test_structures = self.construct_dataset(test_structures, test_targets)


    def construct_dataset(self, structures, targets):
        data_atoms = []
        label = targets
        for _id in tqdm.tqdm(structures.index):
            atoms=AseAtomsAdaptor.get_atoms(structures[_id])
            # set the atomic numbers, positions, and cell
            atom = torch.Tensor(atoms.get_atomic_numbers())
            positions = torch.Tensor(atoms.get_positions())
            cell = torch.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
            natoms = positions.shape[0]

            # put the minimum data in torch geometric data object
            data = Data(
                pos=positions,
                cell=cell,
                atomic_numbers=atom,
                natoms=natoms,
            )

            # calculate energy
            if targets is None:
                data.metadata = None
            else:
                data.metadata = label[_id]
            data_atoms.append(data)
        return data_atoms


    @property
    def trainloader(self):
        return DataLoader(
            self.train_graph_list,
            batch_size=self.config["optim"]["batch_size"],
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )

    @property
    def validloader(self):
        return DataLoader(
            self.test_graph_list,
            batch_size=self.config["optim"]["batch_size"],
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )
    
    def testloader(self, data):
        return DataLoader(
            data,
            batch_size=self.config["optim"]["batch_size"],
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )



# class GemNetFullStructFolds:
#     def __init__(self, 
#             train_structures,
#             train_targets,
#             test_structures,
#             test_targets,
#             configs,
#             graph_construction_config=None,
#             ):

#         self.graph_construction_config = graph_construction_config
#         self.config = configs
        

#         self.train_graph_list = self.prepare(
#             self.get_ase_atoms(train_structures),
#             train_targets,
#             )

#         self.test_graph_list = self.prepare(
#             self.get_ase_atoms(test_structures),
#             test_targets
#         )

#     def get_ase_atoms(self, x):
#         return list(map(AseAtomsAdaptor.get_atoms, x))
    
#     # @cache(name="graph_list_cache")
#     def prepare(self, atoms, targets):
#         print(targets)
#         a2g = AtomsToGraphs(**self.graph_construction_config)
#         if targets is not None:
#             targets = targets.to_frame()
#         return a2g.convert_all(atoms, metadata_collection=targets)

#     def _data_list_collater(self, data_list):
#         batch = Batch.from_data_list(data_list)
#         n_neighbors = []
#         for i, data in enumerate(data_list):
#             n_index = data.edge_index[1, :]
#             n_neighbors.append(n_index.shape[0])
#         batch.neighbors = torch.tensor(n_neighbors)
#         return batch

#     @property
#     def trainloader(self):
#         return DataLoader(
#             self.train_graph_list,
#             batch_size=self.config["optim"]["batch_size"],
#             collate_fn=self._data_list_collater,
#             num_workers=0,#self.config["optim"]["num_workers"],
#             pin_memory=True,
#             shuffle=True,
#         )

#     @property
#     def validloader(self):
#         return DataLoader(
#             self.test_graph_list,
#             batch_size=self.config["optim"]["batch_size"],
#             collate_fn=self._data_list_collater,
#             num_workers=0, #self.config["optim"]["num_workers"],
#             pin_memory=True,
#             shuffle=False,
#         )
    
#     def testloader(self, data):
#         return DataLoader(
#             data,
#             batch_size=self.config["optim"]["batch_size"],
#             collate_fn=self._data_list_collater,
#             num_workers=0, #self.config["optim"]["num_workers"],
#             pin_memory=True,
#             shuffle=False,
#         )
class GemNetFullStruct:
    def __init__(self, config):
        self.config = config

        self._data = pd.read_pickle(self.config["dataset_dir"])
        # Add structure length
        self._data["len_struct"] = self._data["initial_structure"].apply(
            lambda x: len(x.cart_coords)
        )
        self._atoms = list(map(AseAtomsAdaptor.get_atoms, self._data.initial_structure))

        self.graph_list = self.prepare()

    @cache(name="graph_list_cache")
    def prepare(self):
        a2g = AtomsToGraphs()
        return a2g.convert_all(self._atoms, metadata_collection=self._data)

    def _data_list_collater(self, data_list):
        batch = Batch.from_data_list(data_list)
        n_neighbors = []
        for i, data in enumerate(data_list):
            n_index = data.edge_index[1, :]
            n_neighbors.append(n_index.shape[0])
        batch.neighbors = torch.tensor(n_neighbors)
        return batch

    @property
    def trainloader(self):
        return DataLoader(
            self.graph_list,
            batch_size=self.config["optim"]["batch_size"],
            collate_fn=self._data_list_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            shuffle=True,
        )

    @property
    def validloader(self):
        return DataLoader(
            self.graph_list,
            batch_size=self.config["optim"]["batch_size"],
            collate_fn=self._data_list_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            shuffle=False,
        )
