from omegaconf import OmegaConf
from pathlib import Path

class Config:
    def __init__(self, name: str, parent_path=None):
        if not parent_path:
            parent_path = Path(*Path(__file__).resolve().parts[:-3], 'configs', 'gemnet')

        self.path = Path(parent_path).joinpath(
            name if name.endswith('.yaml') else f'{name}.yaml')
        self.config = OmegaConf.load(self.path)
        
