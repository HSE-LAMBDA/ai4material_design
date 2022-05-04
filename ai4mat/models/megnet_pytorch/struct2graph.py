from pymatgen.optimization.neighbors import find_points_in_spheres
import numpy as np
from torch_geometric.data import Data
import torch


class SimpleCrystalConverter:
    def __init__(
            self,
            atom_converter=None,
            bond_converter=None,
            add_z_bond_coord=False,
            cutoff=5.0
    ):
        self.cutoff = cutoff
        self.atom_converter = atom_converter if atom_converter else DummyConverter()
        self.bond_converter = bond_converter if bond_converter else DummyConverter()
        self.add_z_bond_coord = add_z_bond_coord

    def convert(self, d):
        lattice_matrix = np.ascontiguousarray(np.array(d.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
        cart_coords = np.ascontiguousarray(np.array(d.cart_coords), dtype=float)

        center_indices, neighbor_indices, _, distances = find_points_in_spheres(
            cart_coords, cart_coords, r=self.cutoff, pbc=pbc, lattice=lattice_matrix, tol=1e-8
        )

        _, idxs, counts = np.unique(center_indices, return_index=True, return_counts=True)
        idxs = idxs[counts == 1]
        unique_mask = np.zeros_like(center_indices)
        unique_mask[idxs] = True
        exclude_self = (center_indices != neighbor_indices) | unique_mask

        edge_index = torch.Tensor(np.stack((center_indices[exclude_self], neighbor_indices[exclude_self]))).long()
        if torch.numel(edge_index) == 0:
            raise "shit"

        x = torch.Tensor(self.atom_converter.convert(np.array([i.specie.Z for i in d]))).long()

        distances_preprocessed = distances[exclude_self]
        if self.add_z_bond_coord:
            z_coord_diff = np.abs(cart_coords[edge_index[0], 2] - cart_coords[edge_index[1], 2])
            distances_preprocessed = np.stack(
                (distances_preprocessed, z_coord_diff), axis=0
            )

        edge_attr = torch.Tensor(self.bond_converter.convert(distances_preprocessed))
        state = getattr(d, "state", None) or [[0.0, 0.0]]
        y = d.y if hasattr(d, "y") else 0
        bond_batch = torch.Tensor([0 for _ in range(edge_index.shape[1])]).long()

        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, state=torch.Tensor(state), y=y, bond_batch=bond_batch
        )

    def __call__(self, d):
        return self.convert(d)


class DummyConverter:
    def convert(self, d):
        return d.reshape((-1, 1))


class GaussianDistanceConverter:
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        self.centers = centers
        self.sigma = sigma

    def convert(self, d):
        return np.exp(
            -((d.reshape((-1, 1)) - self.centers.reshape((1, -1))) / self.sigma) ** 2
        )


class FlattenGaussianDistanceConverter(GaussianDistanceConverter):
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        super().__init__(centers, sigma)

    def convert(self, d):
        res = []
        for arr in d:
            res.append(super().convert(arr))
        return np.hstack(res)
