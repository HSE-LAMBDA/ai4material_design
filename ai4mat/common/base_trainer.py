import datetime
import torch

from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path
import wandb

    

class Trainer(ABC):
    """"""

    def __init__(
        self,
        name: str,
        run_id,
        model,
        dataset,
        optimizers,
        # scheduler: Optional[Dict[scheduler, kwargs]] = None,
        log_freq=10,
        checkpointing_freq=10,
        logger='wandb',
        rng_seed: int =666,
        use_gpus: Union[int, None] = None,
        run_dir: Optional[Path]=None,
        slurm: dict={},
        **kwargs
    ):
        self.name = name
        self.run_id = run_id
        self.model = model
        self.dataset = dataset
        self.optimizers = optimizers
        self.log_freq = log_freq
        self.checkpointing_freq = checkpointing_freq
        self.logger = logger
        self.run_dir = run_dir
        self.rng_seed = rng_seed
        self.slurm = slurm
        self._step = 0
        self.epoch = 0
        
    
        if use_gpus is not None:
            self.device = f'cuda:{use_gpus}'
        else:
            self.device = 'cpu'

  

        if self.run_dir and not Path(self.run_dir).exists:
            self.run_dir = Path(self.run_dir).joinpath('checkpoints')
            self.run_dir.mkdir(parents=True, exist_ok=True)
        elif not self.run_dir:
            self.run_dir = Path(str(datetime.datetime.now())).joinpath('checkpoints')
            self.run_dir.mkdir(parents=True, exist_ok=True)


        self.move_to_device()
        self.logged_params = {} 

    def load_state_dict(self, file):
        state_dict = torch.load(Path(self.run_dir).joinpath('checkpoints', f'{self.run_id}_{self.name}.pth'))
        self.model.load_state_dict(state_dict['model'])
        # TODO: implement for list of optimizers 
        if not isinstance(self.optimizers, list):
            self.model.load_state_dict(state_dict['model'])
            self.optimizers.load_state_dict(state_dict['optimizers'])
        if self.ema:
            self.ema.load_state_dict(state_dict['ema'])


    def save_state_dict(self, step):
        state_dict = {
            'model' : self.model.state_dict(),
            'optimizers': self.optimizers.state_dict(),
            'logged_params': self.logged_params
        }
        if self.ema:
            state_dict['ema'] = self.ema.state_dict()

        torch.save(state_dict, self.run_dir.joinpath(f'{self.run_id}_{self.name}_{step}.pth'))
    
    def save(self):
        if self.epoch % self.checkpointing_freq == 0:
                self.save_state_dict(step=self.epoch)
        else:
            self.save_state_dict(step='latest')

    def step(self):
        self._step += 1
        
    def log(self, item, epoch):
        self.epoch = epoch
        self.step()
        if self._step % self.log_freq == 0:
            self.logged_params.update(item)
            wandb.log(item)
        
    @abstractmethod
    def train(self):
        pass
    
    @torch.no_grad()
    def predict(self, *args, **kwargs):
        self.model.eval() 
        if self.ema:
            self.ema.store()
            self.ema.copy_to()
        return self.model(*args, **kwargs)
    
    def move_to_device(self):
        self.model.to(self.device)