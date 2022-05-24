import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MegnetModule, ShiftedSoftplus
from torch_geometric.nn import Set2Set


class MEGNet(nn.Module):
    def __init__(self, edge_input_shape, node_input_shape, state_input_shape):
        super().__init__()
        self.emb = nn.Embedding(95, 16)

        self.embedded = True if node_input_shape is None else False
        if self.embedded:
            node_input_shape = 16

        self.m1 = MegnetModule(edge_input_shape, node_input_shape, state_input_shape, inner_skip=True)
        self.m2 = MegnetModule(32, 32, 32)
        self.m3 = MegnetModule(32, 32, 32)
        self.se = Set2Set(32, 1)
        self.sv = Set2Set(32, 1)
        self.hiddens = nn.Sequential(
            nn.Linear(160, 32),
            ShiftedSoftplus(),
            nn.Linear(32, 16),
            ShiftedSoftplus(),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        if self.embedded:
            x = self.emb(x).squeeze()
        else:
            x = x.float()

        x, edge_attr, state = self.m1(x, edge_index, edge_attr, state, batch, bond_batch)
        x, edge_attr, state = self.m2(x, edge_index, edge_attr, state, batch, bond_batch)
        x, edge_attr, state = self.m3(x, edge_index, edge_attr, state, batch, bond_batch)
        x = self.sv(x, batch)
        edge_attr = self.se(edge_attr, bond_batch)

        tmp_shape = x.shape[0] - edge_attr.shape[0]
        edge_attr = F.pad(edge_attr, (0, 0, 0, tmp_shape), value=0.0)

        tmp = torch.cat((x, edge_attr, state), 1)
        out = self.hiddens(tmp)
        return out
