import pandas as pd
import torch
import numpy as np

from tqdm import trange
from ai4mat.common.base_trainer import Trainer
from ai4mat.models.megnet_pytorch.megnet_pytorch import MEGNet
from ai4mat.data.gemnet_dataloader import GemNetFullStruct, GemNetFullStructFolds
from ai4mat.models.megnet_pytorch.struct2graph import SimpleCrystalConverter, GaussianDistanceConverter
from torch_geometric.loader import DataLoader


class MEGNetPyTorchTrainer(Trainer):
    def __init__(
            self,
            train_structures: pd.Series,  # series of pymatgen object
            train_targets: pd.Series,  # series of scalars
            test_structures: pd.Series,  # series of pymatgen object
            test_targets: pd.Series,  # series of scalars
            configs: dict,
            gpu_id: int,
            **kwargs,
    ):

        self.model = MEGNet()

        self.config = configs
        if train_structures is not None and train_targets is not None:
            converter = SimpleCrystalConverter(bond_converter=GaussianDistanceConverter())
            self.structures = [converter.convert(s) for s in train_structures]
            print(self.structures)
            self.target_name = train_targets.name

            self.trainloader = DataLoader(
                self.structures,
                batch_size=32,
                shuffle=False,
            )
        else:
            self.trainloader = GemNetFullStruct(self.config).trainloader

        super().__init__(
            run_id=1,
            name="test",
            model=self.model,
            dataset=self.trainloader,
            optimizers=torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"]["optimizer_params"],
            ),
            use_gpus=None,
        )
        self.save_checkpoint = kwargs['save_checkpoint']

        if self.config["optim"]["scheduler"].lower() == "ReduceLROnPlateau".lower():
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers,
                # mode=self.config["optim"]["mode"],
                # factor=self.config["optim"]["factor"],
                # patience=self.config["optim"]["patience"],
                # verbose=True,
            )

    def train(self):
        for epoch in trange(self.config["optim"]["max_epochs"]):
            print(f'=========== {epoch} ==============')
            print(self.trainloader.__len__(), self.device)
            batch_loss = []
            for i, item in enumerate(self.trainloader):
                print("item", item)
                _loss = []
                _grad_norm = []
                item = item.to(self.device)
                out = self.model(
                    item.x, item.edge_index, item.edge_attr, item.state, item.batch, item.bond_batch
                ).squeeze()
                loss = torch.nn.functional.l1_loss(out.view(-1), getattr(item, 'y'))
                loss.backward()

                self.optimizers.step()
                _loss.append(loss.detach().cpu().numpy())
                self.optimizers.zero_grad()

                batch_loss.append(np.mean(_loss))

            if self.save_checkpoint:
                self.save()
            self.scheduler.step(loss)
            torch.cuda.empty_cache()
            print(
                f"Epoch: {epoch},  Loss: {np.mean(_loss)}, Grad_norm: {np.mean(_grad_norm)}"
            )

    def predict_structures(self, structures):
        data_list = self.structures.construct_dataset(structures, targets=None)
        results = []
        for item in self.structures.testloader(data_list):
            with torch.no_grad():
                results.append(self.model(item.to(self.device)))
        return torch.concat(results).cpu().detach().numpy()
