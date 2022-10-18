import torch
import torch.nn as nn
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    aggr
)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([2.])), requires_grad=False)

    def forward(self, x):
        return self.sp(x) - self.shift


class MegnetModule(MessagePassing):
    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 inner_skip=False,
                 embed_size=32,
                 vertex_aggregation="mean",
                 global_aggregation="mean",
                 ):
        """
        Parameters
        ----------
        edge_input_shape: size of edge features'
        node_input_shape: size of node features'
        state_input_shape: size of global state features'
        inner_skip: use inner or outer skip connection
        embed_size: embedding and output size
        """
        if vertex_aggregation == "lin":
            vertex_aggregation = aggr.MultiAggregation(
                ['mean', 'sum', 'max'],
                mode='proj',
                mode_kwargs={
                    'in_channels': embed_size,
                    'out_channels': embed_size,
                },
            )
        super().__init__(aggr=vertex_aggregation)

        if global_aggregation == "mean":
            self.global_aggregation = global_mean_pool
        elif global_aggregation == "sum":
            self.global_aggregation = global_add_pool
        elif global_aggregation == "max":
            self.global_aggregation = global_max_pool
        else:
            raise ValueError("Unknown global aggregation type")

        self.inner_skip = inner_skip
        self.phi_e = nn.Sequential(
            nn.Linear(4 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        self.phi_u = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        self.phi_v = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        self.preprocess_e = nn.Sequential(
            nn.Linear(edge_input_shape, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        self.preprocess_v = nn.Sequential(
            nn.Linear(node_input_shape, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        self.preprocess_u = nn.Sequential(
            nn.Linear(state_input_shape, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
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
        u_v = self.global_aggregation(x, batch)
        u_e = self.global_aggregation(edge_attr, bond_batch, batch.max().item() + 1)
        state = self.phi_u(torch.cat((u_e, u_v, state), 1))
        return x + x_skip, edge_attr + edge_attr_skip, state + state_skip

    def message(self, x_i, x_j, edge_attr):
        return edge_attr

    def update(self, inputs, x, state, batch):
        return self.phi_v(torch.cat((inputs, x, state[batch, :]), 1))

    def edge_update(self, x_i, x_j, edge_attr, state, bond_batch):
        return self.phi_e(torch.cat((x_i, x_j, edge_attr, state[bond_batch, :]), 1))
