from typing import List, Optional
from itertools import groupby
from operator import attrgetter
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import pathlib
import wandb

from tqdm import trange, tqdm
from ai4mat.common.base_trainer import Trainer
from ai4mat.models.megnet_pytorch.megnet_on_structures import MEGNetOnStructures
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from ai4mat.models.megnet_pytorch.utils import Scaler
from ai4mat.models.loss_functions import MSELoss, MAELoss
from joblib import Parallel, delayed

class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(
        self,
        dataset: List[Data],
    ):
        class_data = [list(g) for _, g in groupby(dataset, attrgetter("weight"))]
        minority_class, majority_class = sorted(class_data, key=len)
        assert len(class_data) == 2, "Only support binary classes are supported"
        weights = list(map(attrgetter('weight'), dataset))
        return super().__init__(weights, len(majority_class) * 2, replacement=True)

class MEGNetPyTorchTrainer(Trainer):
    def __init__(
            self,
            train_data: list,
            test_data: list,
            target_name: str,
            configs: dict,
            gpu_id: int,
            save_checkpoint: bool,
            n_jobs: int = -1,
            minority_class_upsampling: bool = False,
            checkpoint_path: Optional[Path] = None,
    ):
        self.config = configs
        self.minority_class_upsampling = minority_class_upsampling
        self.ema = False

        if gpu_id is None:
            device = 'cpu'
        else:
            device = f'cuda:{gpu_id}'
        self.megnet = MEGNetOnStructures(self.config, n_jobs=n_jobs, device=device)
        self.model = self.megnet.model
        self.converter = self.megnet.converter
        self.scaler = self.megnet.scaler

        print("converting data")
        self.train_structures = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.converter.convert)(s) for s in tqdm(train_data))
        self.test_structures = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.converter.convert)(s) for s in tqdm(test_data))
        self.scaler.fit(self.train_structures)
        self.target_name = target_name
        self.trainloader = DataLoader(
            self.train_structures,
            batch_size=self.config["model"]["train_batch_size"],
            shuffle=False if minority_class_upsampling else True,
            num_workers=0,
            sampler=ImbalancedSampler(self.train_structures) if minority_class_upsampling else None,
        )

        self.testloader = DataLoader(
            self.test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
            num_workers=0
        )

        super().__init__(
            run_id=None,
            name="",
            model=self.model,
            dataset=self.trainloader,
            run_dir=pathlib.Path().resolve(),
            optimizers=torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
            ),
            use_gpus=gpu_id,  # must be changed ol local notebook !!!
            checkpoint_path=checkpoint_path
        )
        self.save_checkpoint = save_checkpoint

        if self.config["optim"]["scheduler"].lower() == "ReduceLROnPlateau".lower():
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers,
                factor=self.config["optim"]["factor"],
                patience=self.config["optim"]["patience"],
                threshold=self.config["optim"]["threshold"],
                min_lr=self.config["optim"]["min_lr"],
                verbose=True,
            )

    def train(self):
        wandb.define_metric("epoch")
        wandb.define_metric(f"{self.target_name} test_loss_per_epoch", step_metric='epoch')
        wandb.define_metric(f"{self.target_name} train_loss_per_epoch", step_metric='epoch')

        for epoch in trange(self.config["model"]["epochs"]):
            print(f'=========== {epoch} ==============')
            print(len(self.trainloader), self.device)
            print(self.target_name)

            batch_loss = []
            total_train = []
            self.model.train(True)
            
            for i, batch in enumerate(self.trainloader):
                batch = batch.to(self.device)
                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                ).squeeze()
                
                loss = MSELoss(
                    self.scaler.transform(batch.y),
                    preds, 
                    weights=(None if self.minority_class_upsampling else batch.weight), 
                    reduction='mean'
                )
                loss.backward()

                self.optimizers.step()
                self.optimizers.zero_grad()

                batch_loss.append(loss.to("cpu").data.numpy())
                total_train.append(
                    MAELoss(
                        self.scaler.inverse_transform(preds),
                        batch.y,
                        weights=(None if self.minority_class_upsampling else batch.weight), 
                        reduction='sum'
                    ).to('cpu').data.numpy()
                )

            total = []
            self.model.train(False)
            with torch.no_grad():
                for batch in self.testloader:
                    batch = batch.to(self.device)

                    preds = self.model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                    ).squeeze()

                    total.append(
                        MAELoss(
                            self.scaler.inverse_transform(preds),
                            batch.y,
                            weights=batch.weight, 
                            reduction='sum'
                        ).to('cpu').data.numpy()
                    )

            if len(self.test_structures) > 0:
                cur_test_loss = sum(total) / len(self.test_structures)
            else:
                cur_test_loss = None
            cur_train_loss = sum(total_train) / len(self.train_structures)
            self.scheduler.step(cur_train_loss)

            if self.save_checkpoint:
                self.save()

            torch.cuda.empty_cache()

            wandb.log({
                f'{self.target_name} test_loss_per_epoch': cur_test_loss,
                f'{self.target_name} train_loss_per_epoch': cur_train_loss,
                'epoch': epoch,
            })

            print(
                f"{self.target_name} Epoch: {epoch}, train loss: {cur_train_loss}, test loss: {cur_test_loss}"
            )

    def predict_test_structures(self):
        results = []
        with torch.no_grad():
            for batch in self.testloader:
                batch = batch.to(self.device)
                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                )
                results.append(self.scaler.inverse_transform(preds))
        if len(results) > 0:
            return torch.concat(results).to('cpu').data.numpy().reshape(-1, 1)
        else:
            return np.empty((0, 0))
