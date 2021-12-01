import numpy as np
import torch
from tqdm import trange

import wandb
from ai4mat.common.config import Config
from ai4mat.common.ema import ExponentialMovingAverage
from ai4mat.data.gemnet_dataloader import GemNetFullStruct
from ai4mat.models.base_trainer import Trainer
from ai4mat.models.gemnet.gemnet import GemNetT


class GemNetTrainer(Trainer):
    def __init__(self):
        self.config = Config("gemnet").config
        self.model = GemNetT(**self.config["model"])
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
            use_gpus=True,
        )

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
        for epoch in trange(self.config["optim"]["max_epochs"]):
            _loss = []
            _grad_norm = []
            for item in self.trainloader:
                item = item.to(self.device)
                out = self.model(item)
                loss = torch.nn.functional.l1_loss(out.view(-1), item.homo)
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

                self.log({"loss": np.mean(_loss), "grad_norm": np.mean(_grad_norm)}, epoch)
            self.save()
            self.scheduler.step(loss)
            torch.cuda.empty_cache()
            print(
                f"Epoch: {epoch},  Loss: {np.mean(_loss)}, Grad_norm: {np.mean(_grad_norm)}"
            )


if __name__ == "__main__":
    wandb.init(project="GemNet", entity="lambda-hse")
    trainer = GemNetTrainer()
    trainer.train()
