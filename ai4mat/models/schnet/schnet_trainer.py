from typing import Optional
import torch
import pandas as pd
from tqdm import tqdm, trange
import wandb

from ai4mat.common.base_trainer import Trainer
from torch_geometric.nn.models.schnet import SchNet
from ai4mat.data.gemnet_dataloader import GemNetFullStructFolds
from ai4mat.models.loss_functions import MAELoss


class SchNetTrainer(Trainer):
    def __init__(
        self,
        train_structures: pd.Series,
        train_targets: pd.Series,
        test_structures: pd.Series,
        test_targets: pd.Series,
        configs: dict,
        gpu_id: Optional[int],
        target_name: Optional[str]
    ):
        self.config = configs
        self.target_name = target_name
        self.structures = GemNetFullStructFolds(
            train_structures,
            train_targets,
            test_structures,
            test_targets,
            configs=self.config,
        )
        self.train_loader = self.structures.trainloader
        self.test_loader = self.structures.testloader(self.structures.test_structures)

        self.model = SchNet(**self.config["model"])
        super().__init__(
            run_id=None,
            name=None,
            model=self.model,
            dataset=self.train_loader,
            checkpoint_path=None,
            optimizers=torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"]["optimizer_params"],
            ),
            use_gpus=gpu_id,
        )
        self.move_to_device()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizers,
            epochs=self.config["optim"]["max_epochs"],
            steps_per_epoch=len(self.train_loader),
            max_lr=self.config["optim"]["max_lr"],
        )

    def train(self):
        wandb.define_metric(f"{self.target_name} train_loss_per_batch", step_metric="batch")
        wandb.define_metric(f'{self.target_name} test_loss_per_epoch', step_metric="epoch")
        batch_number = 0
        for epoch in trange(self.config["optim"]["max_epochs"]):
            self.model.train(True)
            for item in tqdm(self.train_loader):
                item = item.to(self.device)
                out = self.model(item.atomic_numbers.long(), item.pos, item.batch)
                loss = torch.nn.functional.l1_loss(out.view(-1), item.metadata)
                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                self.scheduler.step()
                wandb.log({f"{self.target_name} train_loss_per_batch": loss.item(), "batch": batch_number})
                batch_number += 1
            torch.cuda.empty_cache()
            self.model.train(False)
            test_loss_sum_per_batch = []
            with torch.no_grad():
                for batch in self.test_loader:
                    batch = batch.to(self.device)
                    preds = self.model(batch.atomic_numbers.long(), batch.pos, batch.batch).view(-1)
                    test_loss_sum_per_batch.append(
                        MAELoss(
                            preds,
                            batch.metadata,
                            weights=batch.weight, 
                            reduction='sum'
                        ).detach().cpu().numpy()
                    )
            wandb.log({
                f'{self.target_name} test_loss_per_epoch': sum(test_loss_sum_per_batch) / len(self.test_loader.dataset),
                "epoch": epoch})

    def predict_structures(self, structures):
        data_list = self.structures.construct_dataset(structures, targets=None)
        results = []
        for item in self.structures.testloader(data_list):
            item = item.to(self.device)
            with torch.no_grad():
                results.append(self.model(item.atomic_numbers.long(), item.pos, item.batch))
        return torch.concat(results).detach().cpu().numpy()