from os import sched_getscheduler
import torch
from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path
import wandb

class Trainer(ABC):
    """
    Can be introspected by Trainer.__abstractmethods__
    """

    def __init__(
        self,
        name: str,
        id,
        model,
        dataset,
        optimizers,
        # scheduler: Optional[Dict[scheduler, kwargs]] = None,
        log_freq=100,
        logger='wandb',
        rng_seed: int =666,
        use_gpus: Union[bool, int]=False,
        run_dir: Optional[Path]=None,
        slurm: dict={},
        **kwargs
    ):
        self.name = name
        self.id = id
        self.model = model
        self.dataset = dataset
        self.optimizers = optimizers
        self.log_freq = log_freq
        self.logger = logger
        self.run_dir = run_dir
        self.rng_seed = rng_seed
        self.slurm = slurm
        self.device = 'cpu' 
        self.step = 0

        if use_gpus:
            if isinstance(use_gpus, bool):
                self.device = 'cuda'
#           if isinstance(use_gpus, int):
#               self.device = f'cuda:{use_gpus}'
        

        self.move_to_device()
        self.logged_params = {} 

    def log(self, item):
        if self.step % self.log_freq == 0:
            self.logged_params |= item
            wandb.log(item)
        
    def load_state_dict(self, file):
        state_dict = torch.load(Path(self.run_dir).joinpath('checkpoints', f'{self.id}_{self.name}.pth'))
        self.model.load_state_dict(state_dict['model'])
        # TODO: implement for list of optimizers 
        if not isinstance(self.optimizers, list):
            self.model.load_state_dict(state_dict['model'])
            self.optimizers.load_state_dict(state_dict['optimizers'])
        if self.ema:
            self.ema.load_state_dict(state_dict['ema'])


    def save_state_dict(self):
        state_dict = {
            'model' : self.model.state_dict(),
            'optimizers': self.optimizers.state_dict(),
            'logged_params': self.logged_params
        }
        if self.ema:
            state_dict |= {'ema': self.ema.state_dict()}

        torch.save(state_dict, Path(self.run_dir).joinpath('checkpoints', f'{self.id}_{self.name}.pth'))

    def train(self):
        ...
        # for epoch in trange(self.config["optim"]["max_epochs"]):
        #     loss = self.minibatch_loop()

        #     print(f'Epoch: {epoch},  Loss: {loss}')
    
    # @abstractmethod
    # def minibatch_loop(self):
    #     pass
    
    @torch.no_grad()
    def predict(self, *args, **kwargs):
        self.model.eval() 
        if self.ema:
            self.ema.store()
            self.ema.copy_to()
        return self.model(*args, **kwargs)

    
    # @torch.no_grad()
    # def validate(self, split):
    #     self.model.eval()
        
    def move_to_device(self):
        self.model.to(self.device)