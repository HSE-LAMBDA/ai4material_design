import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()

    def forward(self, x):
        return self.sp(x) - torch.log(torch.tensor([2.]))


class MegnetModule(MessagePassing):
    def __init__(self, edge_input_shape, node_input_shape, state_input_shape, inner_skip=False):
        super().__init__(aggr="mean")
        self.inner_skip = inner_skip
        self.phi_e = nn.Sequential(
            nn.Linear(128, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 32),
            ShiftedSoftplus(),
        )

        self.phi_u = nn.Sequential(
            nn.Linear(96, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 32),
            ShiftedSoftplus(),
        )

        self.phi_v = nn.Sequential(
            nn.Linear(96, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 32),
            ShiftedSoftplus(),
        )

        self.preprocess_e = nn.Sequential(
            nn.Linear(edge_input_shape, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 32),
            ShiftedSoftplus(),
        )

        self.preprocess_v = nn.Sequential(
            nn.Linear(node_input_shape, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 32),
            ShiftedSoftplus(),
        )

        self.preprocess_u = nn.Sequential(
            nn.Linear(state_input_shape, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 32),
            ShiftedSoftplus(),
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        if not self.inner_skip:
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state

            x = self.preprocess_v(x)
            edge_attr = self.preprocess_e(edge_attr)
            state = self.preprocess_u(state)
        else:
            x = self.preprocess_v(x)
            edge_attr = self.preprocess_e(edge_attr)
            state = self.preprocess_u(state)

            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state

        if torch.numel(bond_batch) > 0:
            edge_attr = self.edge_updater(
                edge_index=edge_index, x=x, edge_attr=edge_attr, state=state, bond_batch=bond_batch
            )
        x = self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_attr, state=state, batch=batch
        )
        u_v = global_mean_pool(x, batch)
        u_e = global_mean_pool(edge_attr, bond_batch, batch.max().item() + 1)
        state = self.phi_u(torch.cat((u_e, u_v, state), 1))
        return x + x_skip, edge_attr + edge_attr_skip, state + state_skip

    def message(self, x_i, x_j, edge_attr):
        return edge_attr

    def update(self, inputs, x, state, batch):
        return self.phi_v(torch.cat((inputs, x, state[batch, :]), 1))

    def edge_update(self, x_i, x_j, edge_attr, state, bond_batch):
        return self.phi_e(torch.cat((x_i, x_j, edge_attr, state[bond_batch, :]), 1))