import torch
import numpy as np
from se3_transformer_pytorch import SE3Transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import ase.io
from collections import defaultdict, namedtuple
import torch.autograd as auto
from math import sqrt
from itertools import product
import sys
sys.path.append("./se3_trans")
import torch
import mendeleev
from ase.visualize import view
from ase.visualize.plot import plot_atoms
import random
import pickle

from copy import deepcopy

torch.set_default_dtype(torch.float64) # works best in float64?
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from IPython.display import clear_output


# %matplotlib widget

class Data(Dataset):
    def __init__(self, path: str, pickle_path='data.pickle', normalize=True):
        self.pickle_path = pickle_path
        self.normalize = normalize
        self.mean_std = []
        self.data = []
        try:
            with open(self.pickle_path, 'rb') as f:
                self.data = pickle.load(f)
                
        except FileNotFoundError:
            print('File not found processing the data')
            self.data_raw = self.get_gpaw_trajectories(path)
            self.setup()

    def setup(self):
        for id, atoms_ in self.data_raw.items():
            for num, data in enumerate(atoms_):
                for i, atoms in enumerate(data[:-2]):
                    data_dict = {
                        'input': {
                            'coord':    self.tensor(atoms.positions),
                            'energy':   torch.tensor(atoms.get_potential_energy()),
                            'force':    self.tensor(atoms.get_forces()),
                            'features': self.tensor([self.compute_species_params(int(n)) for n in atoms.get_atomic_numbers()])
                                              }, # Randomly choose initial structure 
                        'target': {     # Relaxed structure
                            'coord':    self.tensor(data[-1].positions),
                            'energy':   torch.tensor(data[-1].get_potential_energy()),
                            'force':    self.tensor(data[-1].get_forces()),
                            'features': self.tensor([self.compute_species_params(int(n)) for n in data[-1].get_atomic_numbers()])
                                    },
                        'id': id,
                        'atoms': atoms
                                }
                    self.data.append(data_dict)
                    print(f'ID: {id}, Processed: {num} sample, # Atom {i}')

        # Save the list of proccessed data        
        with open(self.pickle_path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
    
    
    def __getitem__(self, idx):
        return self.data[idx]
            
    def __len__(self):
        return len(self.data)
    
    def compute_species_params(self, charge):
        element = mendeleev.element(charge)
        return [charge, element.atomic_volume, element.group_id, element.period]
                                    
    def tensor(self, x):
        if not self.normalize:
            return torch.from_numpy(x)
        return torch.tensor(self.norm(x))
    
    def norm(self, x):
        std = np.std(x)
        mean = np.mean(x)
        self.mean_std.append((mean, std))
        return (x / (std + 1e-8)) - mean
    
    def denorm(self, x):
        std = np.std(x)
        mean = np.mean(x)
        return (x * std) + mean
        
    def get_gpaw_trajectories(self, defect_db_path:str):
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


data = Data("./datasets/raw_ruslan_202104/new_datasets/defectDB/", pickle_path='data_fixed.pickle')