import os
import itertools
from operator import methodcaller
import numpy as np
import pandas as pd
import argparse
from multiprocessing import Pool
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount

from megnet.models import MEGNetModel
from megnet.utils.preprocessing import StandardScaler
from megnet.data.graph import GaussianDistance

from defect_representation import VacancyAwareStructureGraph, FlattenGaussianDistance


def get_free_gpu():
    nvmlInit()
    return np.argmax([
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
        for i in range(nvmlDeviceGetCount())
    ])


IS_INTENSIVE = {
    "homo": True,
    "formation_energy": False
}

TRAIN_CORE_PATH = r"datasets/train_defects{}.pickle.gzip"
MODEL_PATH_ROOT = os.path.join("models", "MEGNet-defect-only")
class Experiment():
    def __init__(self,
                 target: str,
                 vacancy_only: bool,
                 add_displaced_species: bool,
                 add_bond_z_coord: bool,
                 epochs: int = 1000,
                 ):
        if vacancy_only:
            self.train_path = TRAIN_CORE_PATH.format("_vac_only")
        else:
            self.train_path = TRAIN_CORE_PATH.format("")
        self.name = (f"{'vac_only' if vacancy_only else 'full'}"
                     f"{'_bond_z' if add_bond_z_coord else ''}"
                     f"{'_werespecies' if add_displaced_species else ''}")
        self.target = target
        self.epochs = epochs
        self.add_displaced_species = add_displaced_species
        self.add_bond_z_coord = add_bond_z_coord
        self.model_path = os.path.join(MODEL_PATH_ROOT, self.target, self.name)
        # This parameter is not used in run(), but is saved for reference
        self.vacancy_only = vacancy_only

    def run(self):
        train = pd.read_pickle(self.train_path)
        nfeat_edge_per_dim = 5
        cutoff = 15
        if self.add_bond_z_coord:
            bond_converter = FlattenGaussianDistance(
                np.linspace(0, cutoff, nfeat_edge_per_dim), 0.5)
        else:
            bond_converter = GaussianDistance(
                np.linspace(0, cutoff, nfeat_edge_per_dim), 0.5)
        graph_converter = VacancyAwareStructureGraph(
            bond_converter=bond_converter,
            add_displaced_species=self.add_displaced_species,
            add_bond_z_coord=self.add_bond_z_coord,
            cutoff=cutoff)
        scaler = StandardScaler.from_training_data(train.defect_representation,
                                                   train[self.target],
                                                   is_intensive=IS_INTENSIVE[self.target])
        # TODO(kazeevn) elegant device configration
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(get_free_gpu())
        # TODO(kazeevn) consider embeddings for the atomic numbers
        model = MEGNetModel(nfeat_edge=graph_converter.nfeat_edge*nfeat_edge_per_dim,
                            nfeat_node=graph_converter.nfeat_node,
                            nfeat_global=2,
                            graph_converter=graph_converter,
                            npass=2,
                            target_scaler=scaler)
        model.train(train.defect_representation,
                    train[self.target],
                    epochs=self.epochs,
                    verbose=1)
        model.save_model(self.model_path)


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


TARGETS = ("homo", "formation_energy")
def generate_expetiments():
    param_values = {
        "target": TARGETS,
        "vacancy_only": (True, False),
        "add_displaced_species": (True, False),
        "add_bond_z_coord": (True, False),
        "epochs": [1000],
    }    
    return [Experiment(**params) for params in product_dict(**param_values)]


def main():
    experiments = generate_expetiments()
    # We are light on GPU usage
    with Pool(8) as p:
        p.map(methodcaller("run"), experiments)

    
if __name__ == '__main__':
    main()
