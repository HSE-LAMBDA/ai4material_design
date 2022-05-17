import os
import numpy as np
import pandas as pd

import wandb

import torch
import ase
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from pymatgen.io.ase import AseAtomsAdaptor

torch.multiprocessing.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")
from tqdm import trange

import wandb
from ai4mat.common.base_trainer import Trainer
from ai4mat.models.schnet.schnet import SchNet
from ai4mat.data.gemnet_dataloader import GemNetFullStructFolds


class SchNetTrainer(Trainer):
    def __init__(
        self,
        train_structures,
        train_targets,
        test_structures,
        test_targets,
        configs,
        gpu_id=0,
        **kwargs,
    ):
        self.config = configs
        self.structures = GemNetFullStructFolds(
            train_structures,
            train_targets,
            test_structures,
            test_targets,
            configs=self.config,
        )
        self.train_loader = self.structures.trainloader
        self.test_loader = self.structures.testloader

        self.model = SchNet(**self.config["model"])
        super().__init__(
            run_id=1,
            name="test",
            model=self.model,
            dataset=self.train_loader,
            optimizers=torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"]["optimizer_params"],
            ),
            use_gpus=gpu_id,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizers,
            epochs=self.config["optim"]["max_epochs"],
            steps_per_epoch=len(self.train_loader),
            max_lr=self.config["optim"]["max_lr"],
        )

    def train(self):
        # Define the custom x axis metric
        wandb.define_metric("epoch")
        wandb.define_metric("dataloader_step")
        # Define which metrics to plot against that x-axis
        wandb.define_metric("loss_per_epoch", step_metric="epoch")
        wandb.define_metric("loss", step_metric="dataloader_step")

        for epoch in trange(self.config["optim"]["max_epochs"]):
            print(f"=========== {epoch} ==============")
            print(self.train_loader.__len__(), self.device)
            self.model.train()
            losses = []
            for i, item in enumerate(self.train_loader):
                item = item.to(self.device)
                out = self.model(item)
                loss = torch.nn.functional.l1_loss(out.view(-1), item.metadata)

                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                self.scheduler.step()

                losses.append(loss.item())
                self.log({"loss": loss.item(), "dataloader_step": i}, epoch)
            wandb.log({"loss_per_epoch": np.mean(losses), "epoch": epoch})
            torch.cuda.empty_cache()

    def predict_structures(self, structures):
        data_list = self.structures.construct_dataset(structures, targets=None)
        results = []
        for item in self.structures.testloader(data_list):
            with torch.no_grad():
                results.append(self.model(item.to(self.device)))
        return torch.concat(results).cpu().detach().numpy()


if __name__ == "__main__":
    wandb.init(project="SchNet", entity="lambda-hse")
    trainer = SchNetTrainer()
    trainer.train()
