from functools import cached_property
from operator import imod
import torch
import numpy as np
from torch_geometric.data import Data
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.optimization.neighbors import find_points_in_spheres
import logging
from copy import deepcopy
from pymatgen.core.periodic_table import Element, ElementBase, DummySpecie
from pymatgen.analysis.local_env import CrystalNN
from collections import defaultdict

class MyTensor(torch.Tensor):
    """
    this class is needed to work with graphs without edges
    """
    def max(self, *args, **kwargs):
        if torch.numel(self) == 0:
            return 0
        else:
            return torch.max(self, *args, **kwargs)


class SimpleCrystalConverter:
    def __init__(
            self,
            atom_converter=None,
            bond_converter=None,
            add_z_bond_coord=False,
            add_eos_features=True,
            cutoff=5.0,
            ignore_state=False,
    ):
        """
        Parameters
        ----------
        atom_converter: converter that converts pymatgen structure to node features
        bond_converter: converter that converts distances to edge features
        add_z_bond_coord: use z-coordinate feature or no
        cutoff: cutoff radius
        ignore_state: ignore global state. State is normally used to
            contain information about the pristine material
        """
        self.cutoff = cutoff
        self.atom_converter = atom_converter if atom_converter else DummyConverter()
        self.bond_converter = bond_converter if bond_converter else DummyConverter()
        self.add_z_bond_coord = add_z_bond_coord
        self.add_eos_features = add_eos_features
        self.ignore_state = ignore_state

    def convert(self, d):
        lattice_matrix = np.ascontiguousarray(np.array(d.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
        cart_coords = np.ascontiguousarray(np.array(d.cart_coords), dtype=float)

        center_indices, neighbor_indices, _, distances = find_points_in_spheres(
            cart_coords, cart_coords, r=self.cutoff, pbc=pbc, lattice=lattice_matrix, tol=1e-8
        )

        exclude_self = (center_indices != neighbor_indices)

        edge_index = torch.Tensor(np.stack((center_indices[exclude_self], neighbor_indices[exclude_self]))).long()

        x = torch.Tensor(self.atom_converter.convert(d)).long()

        distances_preprocessed = distances[exclude_self]
        if self.add_z_bond_coord:
            z_coord_diff = np.abs(cart_coords[edge_index[0], 2] - cart_coords[edge_index[1], 2])
            distances_preprocessed = np.stack(
                (distances_preprocessed, z_coord_diff), axis=0
            )

        edge_attr = torch.Tensor(self.bond_converter.convert(distances_preprocessed))
        
        if self.add_eos_features:
            eos = EOS(defect_rep=d, bond_converter=self.bond_converter)
            edge_attr = torch.tensor(eos.add_eos_features(edge_attr, center_indices, neighbor_indices, distances), dtype=torch.float)

        if self.ignore_state:
            state = [[0.0, 0.0]]
        else:
            state = getattr(d, "state", None) or [[0.0, 0.0]]
        if len(state[0]) > 2:
            raise NotImplementedError("We currently only support state length of 1 and 2")
        if len(state[0]) == 1:
            state[0].append(state[0][0])
            logging.warning("Tiling state from length 1 to length 2")
        y = d.y if hasattr(d, "y") else 0
        bond_batch = MyTensor(np.zeros(edge_index.shape[1])).long()

        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, state=torch.Tensor(state), y=y, bond_batch=bond_batch
        )

    def __call__(self, d):
        return self.convert(d)


class DummyConverter:
    def convert(self, d):
        return d.reshape((-1, 1))

class EOS:
    def __init__(self, defect_rep, bond_converter):
        self.defect_rep = defect_rep
        
    def add_eos_features(self, edge_attr, center_indices, neighbor_indices, distances):
        edges_eos = []
        edges_eos_parity = []
        for center_index, neighbor_index, distance in zip(center_indices, neighbor_indices, distances):
            if center_index == neighbor_index:
                continue
            center_atom = self.defect_rep[center_index]
            neighbor_atom = self.defect_rep[neighbor_index]
            # Get the distance between the center atom and the nearest neighbor atom
            if center_atom.properties['was'] == neighbor_atom.properties['was']:
                edges_eos.append(np.searchsorted(center_atom.properties['shells'], distance))
            else:
                edges_eos.append(np.isclose(center_atom.properties['shells'], distance, rtol=1, atol=1e-01).argmin())
            edges_eos_parity.append(edges_eos[-1] % 2)

        edges_eos = np.vstack([edges_eos, edges_eos_parity]).T
        return np.hstack([edge_attr, edges_eos])


class GaussianDistanceConverter:
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        self.centers = centers
        self.sigma = sigma

    def convert(self, d):
        return np.exp(
            -((d.reshape((-1, 1)) - self.centers.reshape((1, -1))) / self.sigma) ** 2
        )

    def get_shape(self, eos=False):
        shape = len(self.centers)
        if eos:
            shape += 2
        return shape


class FlattenGaussianDistanceConverter(GaussianDistanceConverter):
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        super().__init__(centers, sigma)

    def convert(self, d):
        res = []
        for arr in d:
            res.append(super().convert(arr))
        return np.hstack(res)

    def get_shape(self, eos=False):
        shape = 2 * len(self.centers)
        if eos:
            shape += 2
        return shape


class AtomFeaturesExtractor:
    def __init__(self, atom_features):
        self.atom_features = atom_features

    def convert(self, structure: Structure):
        if self.atom_features == "Z":
            return np.array(
                [0 if isinstance(i, DummySpecies) else i.Z for i in structure.species]
            ).reshape(-1, 1)
        elif self.atom_features == 'werespecies':
            return np.array([
                [
                    0 if isinstance(i.specie, DummySpecies) else i.specie.Z,
                    i.properties["was"],
                ] for i in structure.sites
            ])
        else:
            raise NotImplementedError

    def get_shape(self):
        if self.atom_features == "Z":
            return 1
        elif self.atom_features == 'werespecies':
            return 2
        else:
            return None
