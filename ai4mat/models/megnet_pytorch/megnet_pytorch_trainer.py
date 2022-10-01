import torch
import numpy as np
import torch.nn.functional as F
import pathlib
import wandb

from tqdm import trange, tqdm
from torch_geometric.loader import DataLoader


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

        self.train_loss = []

        #TODO add FlattenGaussianDistanceConverter compatability
        bond_converter = DummyConverter(self.config['model']['edge_embed_size'])
        atom_converter = AtomFeaturesExtractor(self.config["model"]["atom_features"])

        self.model = MEGNet(
            edge_input_shape=bond_converter.get_shape(),
            node_input_shape=atom_converter.get_shape(),
            state_input_shape=self.config["model"]["state_input_shape"],
            soft_cutoff = model_config['model']['soft_cutoff'],
            embedding_size=self.config['model']['edge_embed_size']
        )
        self.Scaler = Scaler()

        self.converter = SimpleCrystalConverter(
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=self.config["model"]["cutoff"],
            add_z_bond_coord=self.config["model"]["add_z_bond_coord"]
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
                [{'params': [j[1] for j in self.model.named_parameters() if not j[0] == 'm1.soft_cutoff']},
                 {'params': self.model.m1.soft_cutoff, 'lr': self.config['model']['soft_cutoff_lr']}],
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

            cur_train_loss = sum(total_train) / len(self.train_structures)
            self.scheduler.step(cur_train_loss)
            self.train_loss.append(cur_train_loss)

            if self.save_checkpoint:
                self.save()

            torch.cuda.empty_cache()

            wandb.log(
                {f'{self.target_name} train_loss_per_epoch': cur_train_loss, 'epoch': epoch}
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