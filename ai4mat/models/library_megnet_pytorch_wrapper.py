from typing import Union
from pathlib import Path
import os
import pandas as pd
import torch
import wandb
from tqdm import trange

from MEGNetSparse import MEGNetTrainer


def get_megnet_pytorch_predictions(
        train_structures: Union[pd.Series, pd.DataFrame] ,  # series of pymatgen object
        train_targets: Union[pd.Series, pd.DataFrame],  # series of scalars
        train_weights,
        test_structures: Union[pd.Series, pd.DataFrame],  # series of pymatgen object
        test_targets: Union[pd.Series, pd.DataFrame],  # series of scalars
        test_weights,
        target_is_intensive: bool,
        model_params: dict,
        gpu: int,
        checkpoint_path: Path,
        n_jobs,
        minority_class_upsampling: bool,
        save_checkpoints: bool,):
    if not target_is_intensive:
        raise NotImplementedError
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    target_name = train_targets.name
    n_epochs = model_params['model']['epochs']

    model = MEGNetTrainer(model_params, f'cuda:{gpu}' if gpu is not None else 'cpu')

    train_targets = torch.tensor(train_targets.tolist()).float()
    train_structures = train_structures.tolist()
    test_targets = torch.tensor(test_targets.tolist())
    test_structures = test_structures.tolist()

    model.prepare_data(
        train_structures, train_targets, test_structures, test_targets, target_name, train_weights, test_weights
    )

    wandb.define_metric("epoch")
    wandb.define_metric(f"{target_name} test_loss_per_epoch", step_metric='epoch')
    wandb.define_metric(f"{target_name} train_loss_per_epoch", step_metric='epoch')

    for epoch in trange(n_epochs):
        print(f'=========== {epoch} ==============')
        step_mae, _ = model.train_one_epoch()
        test_mae = model.evaluate_on_test(return_predictions=False)

        wandb.log({
            f'{target_name} test_loss_per_epoch': test_mae,
            f'{target_name} train_loss_per_epoch': step_mae,
            'epoch': epoch,
        })

        print(
            f"train loss: {step_mae}, test loss: {test_mae}"
        )

    print('========== predicting ==============')
    _, predictions = model.evaluate_on_test(return_predictions=True)
    return predictions.numpy()
