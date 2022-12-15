import os
from ast import Dict

import numpy as np
import torch

torch.multiprocessing.set_start_method('forkserver', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import math

from tqdm import tqdm, trange

import wandb
from ai4mat.common.base_trainer import Trainer
from ai4mat.common.config import Config
from ai4mat.common.ema import ExponentialMovingAverage
from ai4mat.data.gemnet_dataloader import (GemNetFullStruct,
                                           GemNetFullStructFolds)
from ai4mat.models.gemnet.gemnet import GemNetT
from ai4mat.models.loss_functions import MAELoss
from ..modules.scaling import ScaleFactor
from ..modules.scaling.util import ensure_fitted

class FitScalingMixin:
    def fit_scaling(self):
        try:
            ensure_fitted(self.model)
        except ValueError:
            self.scales()
        
    def scales(self):
        print("Model not fitted yet")
        data_batch = next(iter(self.trainloader)).cuda()
        out = self.model(data_batch)
        _ = torch.nn.functional.l1_loss(out.view(-1), getattr(data_batch, 'metadata'))
        del out, _

        scale_factors = {
        name: module
            for name, module in self.model.named_modules()
        if isinstance(module, ScaleFactor)
        }
        fitted_scale_factors = [
            f"{name}: {module.scale_factor.item():.3f}"
            for name, module in scale_factors.items()
            if module.fitted
        ]
        fitted_scale_factors_str = ", ".join(fitted_scale_factors)
        print(f"Fitted scale factors: [{fitted_scale_factors_str}]")
        unfitted_scale_factors = [
            name for name, module in scale_factors.items() if not module.fitted
        ]
        unfitted_scale_factors_str = ", ".join(unfitted_scale_factors)
        print(f"Unfitted scale factors: [{unfitted_scale_factors_str}]")
        

        for name, scale_factor in scale_factors.items():
            if scale_factor.fitted:
                print(
                    f"{name} is already fitted in the checkpoint, resetting it. {scale_factor.scale_factor}"
                )
            scale_factor.reset_()


        scale_factor_indices: Dict[str, int] = {}
        max_idx = 0

        # initialize all scale factors
        for name, module in scale_factors.items():

            def index_fn(name=name):
                nonlocal max_idx
                assert name is not None
                if name not in scale_factor_indices:
                    scale_factor_indices[name] = max_idx
                    print(f"Scale factor for {name} = {max_idx}")
                    max_idx += 1

            module.initialize_(index_fn=index_fn)

        # single pass through network
        _ = self.model(data_batch)
        del _

        # sort the scale factors by their computation order
        sorted_factors = sorted(
            scale_factors.items(),
            key=lambda x: scale_factor_indices.get(x[0], math.inf),
        )

        print("Sorted scale factors by computation order:")
        for name, _ in sorted_factors:
            print(f"{name}: {scale_factor_indices[name]}")

        # endregion

        # loop over the scale factors in the computation order
        # and fit them one by one
        print("Start fitting")

        for name, module in sorted_factors:
            print(f"Fitting {name}...")
            with module.fit_context_():
                for idx, batch in tqdm(enumerate(iter(self.trainloader))):
                    out = self.model(batch.cuda())
                    _ = torch.nn.functional.l1_loss(out.view(-1), getattr(batch, 'metadata'))
                    del out, _
                    # just use 20 batches for fitting
                    if idx > 20: 
                        break
                stats, ratio, value = module.fit_()

                print(
                    f"Variable: {name}, "
                    f"Var_in: {stats['variance_in']:.3f}, "
                    f"Var_out: {stats['variance_out']:.3f}, "
                    f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
                )

        # make sure all scale factors are fitted
        for name, module in sorted_factors:
            assert module.fitted, f"{name} is not fitted"


class GemNetTrainer(Trainer, FitScalingMixin):
    def __init__(self, 
        train_structures=None,
        train_targets=None,
        test_structures=None,
        test_targets=None,
        configs=None,
        gpu_id=0,
        checkpoint_path=None,
        minority_class_upsampling: bool = False,
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
            self.validloader = self.structures.validloader
        else:
            self.trainloader = GemNetFullStruct(self.config).trainloader
            raise NotImplemented

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
        self.fit_scaling()

    def train(self):
        ensure_fitted(self.model)
        # Define the custom x axis metric
        # wandb.define_metric("epoch")
        # wandb.define_metric("dataloader_step")
        # # Define which metrics to plot against that x-axis
        # wandb.define_metric("loss_per_epoch", step_metric='epoch')
        # wandb.define_metric("loss", step_metric='dataloader_step')
        # wandb.define_metric("grad_norm", step_metric='dataloader_step')
        
        wandb.define_metric(f"{self.target_name} train_loss_per_batch", step_metric="batch")
        wandb.define_metric(f'{self.target_name} test_loss_per_epoch', step_metric="epoch")
        batch_number = 0
        _grad_norm = []
        for epoch in trange(self.config["optim"]["max_epochs"]):
            self.model.train(True)
            for item in tqdm(self.trainloader):
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
                self.ema.update()
                self.optimizers.zero_grad()

                wandb.log({f"{self.target_name} train_loss_per_batch": loss.item(), "batch": batch_number})
                batch_number += 1
            if self.save_checkpoint:
                self.save()
            self.scheduler.step(loss)
            torch.cuda.empty_cache()

            self.model.train(False)
            test_loss_sum_per_batch = []
            with torch.no_grad():
                for batch in self.validloader:
                    batch = batch.to(self.device)
                    preds = self.model(batch).view(-1)
                    test_loss_sum_per_batch.append(
                        MAELoss(
                            preds,
                            batch.metadata,
                            weights=batch.weight, 
                            reduction='sum'
                        ).detach().cpu().numpy()
                    )
            wandb.log({
                f'{self.target_name} test_loss_per_epoch': sum(test_loss_sum_per_batch) / len(self.validloader.dataset),
                "epoch": epoch})

    def predict_structures(self, structures):
        ensure_fitted(self.model)
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
