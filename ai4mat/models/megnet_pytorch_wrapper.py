from lib2to3.pytree import convert
from typing import Union
import pandas as pd
from ai4mat.models.megnet_pytorch.megnet_pytorch_trainer import MEGNetPyTorchTrainer
import os


def set_y(structure, y):
    setattr(structure, "y", y)
    return structure


def get_megnet_pytorch_predictions(
        train_structures: Union[pd.Series, pd.DataFrame] ,  # series of pymatgen object
        train_targets: Union[pd.Series, pd.DataFrame],  # series of scalars
        test_structures: Union[pd.Series, pd.DataFrame],  # series of pymatgen object
        test_targets: Union[pd.Series, pd.DataFrame],  # series of scalars
        target_is_intensive: bool,
        model_params: dict,
        gpu: int,
        checkpoint_path,
        n_jobs,
        ):
    if not target_is_intensive:
        raise NotImplementedError
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    target_name = train_targets.name
    train_targets = train_targets.tolist()
    test_targets = test_targets.tolist()
    train_data = [set_y(s, y) for s, y in zip(train_structures, train_targets)]
    test_data = [set_y(s, y) for s, y in zip(test_structures, test_targets)]

    model = MEGNetPyTorchTrainer(
        train_data,
        test_data,
        target_name,
        configs=model_params,
        gpu_id=gpu,
        save_checkpoint=False,
        n_jobs=n_jobs
    )
    model.train()

    print('========== predicting ==============')
    return model.predict_test_structures()
