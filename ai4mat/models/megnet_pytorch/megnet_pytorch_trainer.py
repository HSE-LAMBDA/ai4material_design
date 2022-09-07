import torch
import numpy as np
import torch.nn.functional as F
import pathlib
import wandb

from tqdm import trange, tqdm
from ai4mat.common.base_trainer import Trainer
from ai4mat.models.megnet_pytorch.megnet_pytorch import MEGNet
from ai4mat.models.megnet_pytorch.struct2graph import SimpleCrystalConverter, GaussianDistanceConverter
from ai4mat.models.megnet_pytorch.struct2graph import FlattenGaussianDistanceConverter, AtomFeaturesExtractor
from torch_geometric.loader import DataLoader
from ai4mat.models.megnet_pytorch.utils import Scaler


class MEGNetPyTorchTrainer(Trainer):
    def __init__(
            self,
            train_data: list,
            test_data: list,
            target_name: str,
            configs: dict,
            gpu_id: int,
            save_checkpoint: bool,
    ):
        self.config = configs

        if self.config["model"]["add_z_bond_coord"]:
            bond_converter = FlattenGaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        else:
            bond_converter = GaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        atom_converter = AtomFeaturesExtractor(self.config["model"]["atom_features"])

        self.model = MEGNet(
            edge_input_shape=bond_converter.get_shape(),
            node_input_shape=atom_converter.get_shape(),
            state_input_shape=self.config["model"]["state_input_shape"]
        )
        self.Scaler = Scaler()

        self.converter = SimpleCrystalConverter(
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=self.config["model"]["cutoff"],
            add_z_bond_coord=self.config["model"]["add_z_bond_coord"]
            add_eos_features=self.config["model"]["add_eos_features"]
        )
        print("converting data")
        self.train_structures = [self.converter.convert(s) for s in tqdm(train_data)]
        self.test_structures = [self.converter.convert(s) for s in tqdm(test_data)]
        self.Scaler.fit(self.train_structures)
        self.target_name = target_name

        self.trainloader = DataLoader(
            self.train_structures,
            batch_size=self.config["model"]["train_batch_size"],
            shuffle=True,
            num_workers=0
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
                loss = F.mse_loss(self.Scaler.transform(batch.y), preds)
                loss.backward()

                self.optimizers.step()
                self.optimizers.zero_grad()

                batch_loss.append(loss.to("cpu").data.numpy())
                total_train.append(
                    F.l1_loss(self.Scaler.inverse_transform(preds), batch.y, reduction='sum').to('cpu').data.numpy()
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
                        F.l1_loss(self.Scaler.inverse_transform(preds), batch.y, reduction='sum').to('cpu').data.numpy()
                    )

            cur_test_loss = sum(total) / len(self.test_structures)
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
                results.append(self.Scaler.inverse_transform(preds))
        return torch.concat(results).to('cpu').data.numpy().reshape(-1, 1)
