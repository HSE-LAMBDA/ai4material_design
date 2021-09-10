import os
import itertools
import time
from operator import methodcaller
import numpy as np
import pandas as pd
import argparse
from multiprocessing import Pool
import pynvml
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
import wandb

from megnet.models import MEGNetModel
from megnet.utils.preprocessing import StandardScaler
from megnet.data.graph import GaussianDistance

from defect_representation import VacancyAwareStructureGraph, FlattenGaussianDistance

IS_INTENSIVE = {
    "homo": True,
    "formation_energy": False,
    "band_gap": True,
    "formation_energy_per_site": True
}

MODEL_PATH_ROOT = os.path.join("models", "MEGNet-defect-only")


def get_free_gpu():
    nvmlInit()
    return np.argmax([
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
        for i in range(nvmlDeviceGetCount())
    ])


class Experiment():
    def __init__(self,
                 data_path: str,
                 folds_path: str,
                 total_folds: int,
                 test_fold: list,
                 target: str,
                 atom_features: str,
                 add_bond_z_coord: bool,
                 epochs: int = 1000,
                 learning_rate: float = 1e-3,
                 supercell_replication = None):
        self.name = (f"fold_{test_fold}"
                     f"{'_bond_z' if add_bond_z_coord else ''}"
                     f"_{atom_features}"
                     f"{'_ss_replication' if supercell_replication else ''}"
                     f"_{epochs}")
        self.target = target
        self.epochs = epochs
        self.atom_features = atom_features
        self.add_bond_z_coord = add_bond_z_coord
        self.model_path = os.path.join(MODEL_PATH_ROOT, self.target, self.name)
        self.learning_rate = learning_rate
        self.supercell_replication = supercell_replication
        self.data_path = data_path
        self.folds_path = folds_path
        self.total_folds = total_folds
        self.test_fold = test_fold
        if epochs is not None and supercell_replication:
            print("Warning: epochs is ignored if supercell_replication is set")


    def run(self, gpu=None):
        data =  pd.read_pickle(self.data_path)
        folds = pd.read_csv(self.folds_path, squeeze=True, index_col="_id")

        train_folds = set(range(self.total_folds)) - set((self.test_fold,))
        train_ids = folds[folds.isin(train_folds)]
        train = data.reindex(index=train_ids.index)

        test_ids = folds[folds == self.test_fold]
        test = data.reindex(index=test_ids.index)

        run = wandb.init(project='ai4material_design',
                         entity=os.environ["WANDB_ENTITY"],
                         config=self.__dict__,
                         reinit=True)
        nfeat_edge_per_dim = 10
        cutoff = 15
        if self.add_bond_z_coord:
            bond_converter = FlattenGaussianDistance(
                np.linspace(0, cutoff, nfeat_edge_per_dim), 0.5)
        else:
            bond_converter = GaussianDistance(
                np.linspace(0, cutoff, nfeat_edge_per_dim), 0.5)
        graph_converter = VacancyAwareStructureGraph(
            bond_converter=bond_converter,
            atom_features=self.atom_features,
            add_bond_z_coord=self.add_bond_z_coord,
            cutoff=cutoff)
        # TODO(kazeevn) do we need to have a separate scaler for each
        # supercell replication?
        # In the intensive case, we seem fine.
        scaler = StandardScaler.from_training_data(train.defect_representation,
                                                   train[self.target],
                                                   is_intensive=IS_INTENSIVE[self.target])
        # TODO(kazeevn) elegant device configration and GPU distribution
        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
        if gpu is None:
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(get_free_gpu())
            except pynvml.nvml.NVMLError_LibraryNotFound:
                pass
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        model = MEGNetModel(nfeat_edge=graph_converter.nfeat_edge*nfeat_edge_per_dim,
                            nfeat_node=graph_converter.nfeat_node,
                            nfeat_global=2,
                            graph_converter=graph_converter,
                            npass=2,
                            target_scaler=scaler,
                            metrics=["mae"],
                            lr=self.learning_rate)
        if self.supercell_replication:
            model = train_with_supercell_replication(
                model,
                train.defect_representation,
                train[self.target],
                test.defect_representation,
                test[self.target],
                is_intensive=IS_INTENSIVE[self.target],
                callbacks=[wandb.keras.WandbCallback(save_model=False)],
                **self.supercell_replication)
        else:
            # We use the same test for monitoring, but do no early stopping
            model.train(train.defect_representation,
                        train[self.target],
                        test.defect_representation,
                        test[self.target],
                        epochs=self.epochs,
                        callbacks=[wandb.keras.WandbCallback(save_model=False)],
                        save_checkpoint=False,
                        verbose=1)
        model.save_model(self.model_path)
        run.finish()


def train_with_supercell_replication(
        model, train_structures, train_target,
        test_structures, test_target,
        callbacks, is_intensive,
        epochs_per_replication_variant,
        replication_iterations,
        max_replications,
        random_seed):

    assert max_replications >= 1
    rng = np.random.RandomState(random_seed)

    for replication_iteration in range(replication_iterations):
        augmented_train = []
        replications = rng.randint(low=1,
                                   high=max_replications + 1,
                                   size=[len(train_structures), 2])
        for structure, replication_params in zip(train_structures, replications):
            augmented_train.append(structure.copy())
            augmented_train[-1].make_supercell([
                replication_params[0],
                replication_params[1],
                1])
            augmented_train[-1].state = structure.state
        if is_intensive:
            augmented_train_target = train_target
        else:
            augmented_train_target = train_target * replication_params.prod(axis=1)
        model.train(
            augmented_train,
            augmented_train_target,
            test_structures,
            test_target,
            epochs=epochs_per_replication_variant,
            callbacks=callbacks,
            save_checkpoint=False,
            verbose=True
        )
    return model


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def generate_paper_experiments():
    params = {
        "atom_features": ("Z",),
        "add_bond_z_coord": (True,),
        "learning_rate": (2e-4,),
        "target": ("formation_energy_per_site",),# "homo", "band_gap"),
        "epochs": (10,),
        "supercell_replication": (None,),
        "total_folds": (8,),
        "test_fold": range(8),
        "data_path": ("datasets/all_structures_defects.pickle.gzip",),
        "folds_path": ("datasets/paper_experiments/inputs/full.csv",)
    }
    return [Experiment(**params) for params in product_dict(**params)]


def run_on_gpu(index_experiment):
    return index_experiment[1].run(gpu=index_experiment[0]%4)


def main():
    parser = argparse.ArgumentParser("Runs MEGENet on sparse defects training")
    parser.add_argument("--gpu-by-index", action="store_true")
    args = parser.parse_args()
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_RUN_GROUP"] = "Defect-only-MEGNet-" + wandb.util.generate_id()
    experiments = generate_paper_experiments()
    # We are light on GPU usage
    with Pool(12) as p:
        if args.gpu_by_index:
            p.map(run_on_gpu, enumerate(experiments))
        else:
            p.map(methodcaller("run"), experiments)


if __name__ == '__main__':
    main()
