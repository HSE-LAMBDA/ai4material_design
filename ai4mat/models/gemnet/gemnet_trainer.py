import os
import numpy as np
import torch

torch.multiprocessing.set_start_method('forkserver', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import trange

import wandb
from ai4mat.common.config import Config
from ai4mat.common.ema import ExponentialMovingAverage
from ai4mat.data.gemnet_dataloader import GemNetFullStruct, GemNetFullStructFolds
from ai4mat.common.base_trainer import Trainer
from ai4mat.models.gemnet.gemnet import GemNetT
from torch.utils.data import DataLoader


class GemNetTrainer(Trainer):
    def __init__(self, 
        train_structures=None,
        train_targets=None,
        test_structures=None,
        test_targets=None,
        configs=None,
        gpu_id=0,
        checkpoint_path=None,
        **kwargs
        ):
        if configs:
            self.config = configs
        else:
            self.config = Config("gemnet").config

        self.model = GemNetT(**self.config["model"], otf_graph=True)
        if train_structures is not None and train_targets is not None:
            self.structures = GemNetFullStructFolds(
                train_structures,
                train_targets,
                test_structures,
                test_targets,
                # max_neigh=200,
                configs=self.config,
                graph_construction_config = {
                    'radius': self.config['model']['cutoff'],
                    'max_neigh': self.config['model']['max_neighbors'],
                    'r_energy': False,
                    'r_forces': False,
                    'r_distances': False,
                    'r_edges': True,
                    'r_fixed': True,
                    'r_homo': False,
                    'r_lumo': False,
                    'r_band_gap': False,
                    'r_other_metadata': True,
                }
                )
            self.target_name = train_targets.name 

            self.trainloader = self.structures.trainloader
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
            # TODO: this need to be managed by the configs
            use_gpus=gpu_id,
            # run_dir=os.environ["WANDB_RUN_GROUP"]
            checkpoint_path=checkpoint_path
        )
        self.save_checkpoint = kwargs['save_checkpoint']
        
        if self.config["optim"]["scheduler"].lower() == "ReduceLROnPlateau".lower():
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers,
                mode=self.config["optim"]["mode"],
                factor=self.config["optim"]["factor"],
                patience=self.config["optim"]["patience"],
                verbose=True,
            )
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), self.config["optim"]["ema_decay"]
        )


    def train(self):

        # Define the custom x axis metric
        wandb.define_metric("epoch")
        wandb.define_metric("dataloader_step")
        # Define which metrics to plot against that x-axis
        wandb.define_metric("loss_per_epoch", step_metric='epoch')
        wandb.define_metric("loss", step_metric='dataloader_step')
        wandb.define_metric("grad_norm", step_metric='dataloader_step')
        
        for epoch in trange(self.config["optim"]["max_epochs"]):
            print(f'=========== {epoch} ==============')
            batch_loss = []
            for i, item in enumerate(self.trainloader):
                _loss = []
                _grad_norm = []
                item = item.to(self.device)
                out = self.model(item)
                loss = torch.nn.functional.l1_loss(out.view(-1), getattr(item, 'metadata'))
                loss.backward()

                ## Grad clipping
                if self.config["optim"]["clip_grad_norm"]:
                    _grad_norm.append(
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config["optim"]["clip_grad_norm"],
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                self.optimizers.step()
                _loss.append(loss.detach().cpu().numpy())
                self.ema.update()
                self.optimizers.zero_grad()

                self.log({"loss": np.mean(_loss), "grad_norm": np.mean(_grad_norm), 'dataloader_step': i}, epoch)
                batch_loss.append(np.mean(_loss))
            wandb.log({'loss_per_epoch': np.mean(batch_loss), 'epoch': epoch})
            if self.save_checkpoint:
                self.save()
            self.scheduler.step(loss)
            torch.cuda.empty_cache()
            print(
                f"Epoch: {epoch},  Loss: {np.mean(_loss)}, Grad_norm: {np.mean(_grad_norm)}"
            )

    def predict_structures(self, structures):
        # data_list = self.structures.prepare(self.structures.get_ase_atoms(structures), targets=None)
        data_list = self.structures.construct_dataset(structures, targets=None)
        results = []
        for item in self.structures.testloader(data_list):
            with torch.no_grad(): 
                results.append(self.model(item.to(self.device)))
        return torch.concat(results).cpu().detach().numpy()
         

if __name__ == "__main__":
    wandb.init(project="GemNet", entity="lambda-hse")
    trainer = GemNetTrainer()
    trainer.train()
