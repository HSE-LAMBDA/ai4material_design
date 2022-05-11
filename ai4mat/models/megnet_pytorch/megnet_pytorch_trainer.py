import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import pathlib

from tqdm import trange
from ai4mat.common.base_trainer import Trainer
from ai4mat.models.megnet_pytorch.megnet_pytorch import MEGNet
from ai4mat.models.megnet_pytorch.struct2graph import SimpleCrystalConverter, GaussianDistanceConverter, FlattenGaussianDistanceConverter
from torch_geometric.loader import DataLoader
from ai4mat.models.megnet_pytorch.utils import Scaler

torch.multiprocessing.set_start_method("forkserver", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")


class MEGNetPyTorchTrainer(Trainer):
    def __init__(
            self,
            train_data: list,
            test_data: list,
            configs: dict,
            gpu_id: int,
            save_checkpoint: bool,
    ):

        self.model = MEGNet()
        self.config = configs
        self.Scaler = Scaler()

        self.converter = SimpleCrystalConverter(bond_converter=GaussianDistanceConverter(), cutoff=0.5)
        self.converter2 = SimpleCrystalConverter(bond_converter=GaussianDistanceConverter(), cutoff=5.0)
        self.train_structures = [self.converter.convert(s) if i % 2 else self.converter2.convert(s) for i, s in enumerate(train_data)]
        self.test_structures = [self.converter.convert(s) for s in test_data]
        self.Scaler.fit(self.train_structures)

        self.trainloader = DataLoader(
            self.train_structures,
            batch_size=self.config["model"]["train_batch_size"],
            shuffle=False,
        )
        # for batch in self.trainloader:
        #     print(batch)
        #     print(batch.bond_batch)
        #     print(batch.bond_batch.shape)
        #     print(batch.batch)
        #     return
        self.testloader = DataLoader(
            self.test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
        )

        super().__init__(
            run_id=1,
            name="test",
            model=self.model,
            dataset=self.trainloader,
            run_dir=pathlib.Path().resolve(),
            optimizers=torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
            ),
            use_gpus=None,
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
        for epoch in trange(self.config["model"]["epochs"]):
            print(f'=========== {epoch} ==============')
            print(len(self.trainloader), self.device)

            batch_loss = []
            self.model.train(True)
            for i, batch in enumerate(self.trainloader):
                batch = batch.to(self.device)
                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                ).squeeze()
                loss = F.mse_loss(self.Scaler.transform(batch.y), preds)
                loss.backward()

                self.optimizers.step()
                self.optimizers.zero_grad()

                batch_loss.append(loss.to("cpu").data.numpy())

            total = []
            self.model.train(False)
            with torch.no_grad():
                for batch in self.testloader:
                    batch = batch.to(self.device)

                    preds = self.model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                    ).squeeze()

                    total.append(
                        F.l1_loss(self.Scaler.inverse_transform(preds), batch.y, reduction='sum') \
                            .to('cpu').data.numpy()
                    )

            self.scheduler.step(sum(total) / len(self.test_structures))

            # if self.save_checkpoint:
            #     self.save()

            torch.cuda.empty_cache()

            print(
                f"Epoch: {epoch}, train loss: {np.mean(batch_loss)}, test loss: {sum(total) / len(self.test_structures)}"
            )

    def predict_test_structures(self):
        results = []
        with torch.no_grad():
            for batch in self.testloader:
                batch = batch.to(self.device)
                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                )
                results.append(self.Scaler.inverse_transform(preds))
        return torch.concat(results).to('cpu').data.numpy().reshape(-1, 1)
