import pytest
from pathlib import Path

try:
    import ai4mat  
except ModuleNotFoundError:
    import sys
    sys.path.append('..')

from ai4mat.models.gemnet.gemnet import GemNetT
from ai4mat.data.gemnet_dataloader import GemNetFullStruct
from ai4mat.common.config import Config

CONFIG = Config('test_gemnet_config', parent_path='.').config

@pytest.fixture(scope='class')
def load_data(request):
    gemnet_loader = GemNetFullStruct(CONFIG)
    data = iter(gemnet_loader.trainloader)
    request.cls.data = data

@pytest.fixture(scope='class')
def load_model(request):
    model = GemNetT(
        num_targets=1,
        cutoff=6.0,
        num_spherical=7,
        num_radial=128,
        num_blocks=3,
        emb_size_atom=16,
        emb_size_edge=16,
        emb_size_trip=16,
        emb_size_rbf=16,
        emb_size_cbf=16,
        emb_size_bil_trip=64,
        num_before_skip=1,
        num_after_skip=2,
        num_concat=1,
        num_atom=3,
        regress_forces=False,
        direct_forces=False,
        scale_file='gemnet_scaling_factors.json'
    )
    request.cls.model = model


@pytest.mark.usefixtures('load_data')
@pytest.mark.usefixtures('load_model')
class TestGemNet:
    def test_GemNetT(self):
        energy = self.model(next(self.data))
        print(energy)
        assert energy is not None and len(energy) == 2