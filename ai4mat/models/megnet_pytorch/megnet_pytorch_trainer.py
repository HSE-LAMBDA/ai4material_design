from typing import List, Optional
from itertools import groupby
from operator import attrgetter
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
from torch_geometric.data import Data

from ai4mat.models.megnet_pytorch.utils import Scaler
from ai4mat.models.loss_functions import weightedMSELoss, weightedMAELoss, MSELoss, MAELoss
from joblib import Parallel, delayed

class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(
        self,
        dataset: List[Data],
    ):
        class_data = [list(g) for _, g in groupby(dataset, key=lambda x: x.weight)]
        minority_class, majority_class = sorted(class_data, key=len)
        assert len(class_data) == 2, "Only support binary classes are supported"
        for group in class_data:
            class_weight = len(group) / len(dataset)
            for data in group:
                data.weight = class_weight
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
        self.converter = SimpleCrystalConverter(
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=self.config["model"]["cutoff"],
            add_z_bond_coord=self.config["model"]["add_z_bond_coord"],
            add_eos_features=(use_eos := self.config["model"].get("add_eos_features", False)),
        )
        self.model = MEGNet(
            edge_input_shape=bond_converter.get_shape(eos=use_eos),
            node_input_shape=atom_converter.get_shape(),
            state_input_shape=self.config["model"]["state_input_shape"],
            vertex_aggregation=self.config["model"]["vertex_aggregation"],
            global_aggregation=self.config["model"]["global_aggregation"],
        )
        self.Scaler = Scaler()

        print("converting data")
        self.train_structures = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.converter.convert)(s) for s in tqdm(train_data))
        self.test_structures = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.converter.convert)(s) for s in tqdm(test_data))
        self.Scaler.fit(self.train_structures)
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
        
        self.MAELoss = MAELoss if minority_class_upsampling else weightedMAELoss
        self.MSELoss = MSELoss if minority_class_upsampling else weightedMSELoss

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
                loss = self.MSELoss(self.Scaler.transform(batch.y), preds, batch.weight, 'mean')
                loss.backward()

                self.optimizers.step()
                self.optimizers.zero_grad()

                batch_loss.append(loss.to("cpu").data.numpy())
                total_train.append(
                    self.MAELoss(self.Scaler.inverse_transform(preds), batch.y, batch.weight, 'sum').to(
                        'cpu').data.numpy()
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
                        self.MAELoss(self.Scaler.inverse_transform(preds), batch.y, batch.weight, reduction='sum') \
                            .to('cpu').data.numpy()
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
