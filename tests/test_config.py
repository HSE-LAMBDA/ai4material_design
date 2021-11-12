import pytest
from pathlib import Path
from ai4mat.common.config import Config

config_dict = {'A': 1, 'B': 2}

def test_config_loading():
    config = Config('test_config.yaml', parent_path=Path(__file__).parent).config
    assert config == config_dict

