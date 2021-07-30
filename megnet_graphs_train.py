import os
import numpy as np
import pandas as pd
import argparse
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount

from megnet.models import MEGNetModel
from megnet.utils.preprocessing import StandardScaler
from megnet.data.graph import GaussianDistance

from defect_representation import VacancyAwareStructureGraph


def get_free_gpu():
    nvmlInit()
    return np.argmax([
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
        for i in range(nvmlDeviceGetCount())
    ])


def main():
    # TODO(kazeevn) elegant device configration
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(get_free_gpu())
    parser = argparse.ArgumentParser(description='Train MEGNet on defect graph.')
    parser.add_argument('--train', type=str, required=True, help='pickled train to use')
    parser.add_argument('--target', type=str, required=True, help='target to predict')
    parser.add_argument('--is-intensive', type=bool, required=True)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    train = pd.read_pickle(args.train)

    nfeat_edge = 10
    gc = VacancyAwareStructureGraph(
        bond_converter=GaussianDistance(np.linspace(0, 15, nfeat_edge), 0.5),
        cutoff=15)
    model = MEGNetModel(nfeat_edge=nfeat_edge,
                        nfeat_node=gc.n_atom_features,
                        nfeat_global=2,
                        graph_converter=gc,
                        npass=2)
    scaler = StandardScaler.from_training_data(train.defect_representation,
                                               train[args.target],
                                               is_intensive=args.is_intensive)
    model.target_scaler = scaler
    model.train(train.defect_representation,
                train[args.target],
                epochs=args.epochs,
                verbose=1)
    model.save_model(os.path.join("models", "MEGNet-defect-only", args.target, args.experiment_name))

    
if __name__ == '__main__':
    main()
