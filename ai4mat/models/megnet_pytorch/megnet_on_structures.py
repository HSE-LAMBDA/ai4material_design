import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader

from ai4mat.models.megnet_pytorch.megnet_pytorch import MEGNet
from ai4mat.models.megnet_pytorch.utils import Scaler
from ai4mat.models.megnet_pytorch.struct2graph import (
    SimpleCrystalConverter, GaussianDistanceConverter,
    FlattenGaussianDistanceConverter, AtomFeaturesExtractor)

class MEGNetOnStructures(torch.nn.Module):
    def __init__(self, config, n_jobs=-1, device='cpu'):
        super().__init__()
        self.config = config        
        if self.config["model"]["add_z_bond_coord"]:
            bond_converter = FlattenGaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        else:
            bond_converter = GaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        atom_converter = AtomFeaturesExtractor(self.config["model"]["atom_features"])
        self.converter = SimpleCrystalConverter(
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=self.config["model"]["cutoff"],
            add_z_bond_coord=self.config["model"]["add_z_bond_coord"],
            add_eos_features=(use_eos := self.config["model"].get("add_eos_features", False)),
        )
        self.model = MEGNet(
            edge_input_shape=bond_converter.get_shape(eos=use_eos),
            node_input_shape=atom_converter.get_shape(),
            embedding_size=self.config['model']['embedding_size'],
            n_blocks=self.config['model']['nblocks'],
            state_input_shape=self.config["model"]["state_input_shape"],
            vertex_aggregation=self.config["model"]["vertex_aggregation"],
            global_aggregation=self.config["model"]["global_aggregation"],
        )
        self.n_jobs = n_jobs
        self.device = device
        self.scaler = Scaler()
    
    def load(self, checkpoint_file_name):
        checkpoint = torch.load(checkpoint_file_name)
        self.model.load_state_dict(checkpoint['model'])
        self.scaler.load_state_dict(checkpoint['scaler'])
    
    def predict_structures(self, sparse_structures):
        test_structures = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self.converter.convert)(s) for s in tqdm(sparse_structures))
        testloader = DataLoader(
            test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
            num_workers=0,
        )
        results = []
        with torch.no_grad():
            for batch in testloader:
                batch = batch.to(self.device)
                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                )
                results.append(self.scaler.inverse_transform(preds))
            return torch.concat(results).to('cpu').data.numpy().reshape(-1, 1)