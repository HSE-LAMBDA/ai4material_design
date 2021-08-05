from copy import deepcopy
from typing import List, Dict, Union
import numpy as np

from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core import Structure
from pymatgen.analysis.local_env import NearNeighbors

from megnet.data.crystal import CrystalGraph
from megnet.data.graph import Converter
from megnet.data.graph import GaussianDistance

# TODO(kazeevn) separate scales per axis, as dZ << dXY usually
class FlattenGaussianDistance(GaussianDistance):
    def convert(self, d: np.ndarray) -> np.ndarray:
        """
        Presumes that the bond features are different distances, e. g.
        the total distances and distance in z axis produced by
        VacancyAwareStructureGraph with add_bond_z_coord.

        Transforms each distance component with 
        megnet.data.graph.GaussianDistance and concatenates the
        resulting vectors to serve as edge features.
        """
        if len(d.shape) != 2:
            return ValueError("Input array must be 2-dimensional. Use plain GaussianDistance.")
        return np.concatenate(
            [super(FlattenGaussianDistance, self).convert(arr) for arr in d.T],
            axis=1)


class VacancyAwareStructureGraph(CrystalGraph):
    def __init__(self,
                 nn_strategy: Union[str, NearNeighbors] = "MinimumDistanceNNAll",
                 atom_converter: Converter = None,
                 bond_converter: Converter = None,
                 cutoff: float = 5.0,
                 atom_features: str = "embed",
                 add_bond_z_coord: bool = True):
        """"
        Args:
         Same as CrystalGraph:
           nn_strategy (str): NearNeighbor strategy
           atom_converter (Converter): atom features converter
           bond_converter (Converter): bond features converter
           cutoff (float): cutoff radius
         Added:
           atom_features (str):
                "embed": preserves the vanilla CrystalGraph behaviour, atomic number for embedding
                "Z": charge as 1-d feature
                "werespecies": vector of [Z, were_Z], where were_Z is the charge of the species that
                occupied the defects site in the pristine material
           add_bond_z_coord: if set, add abs(atom_1.z_coord - atom_2.z_coord) as a bond feature.
                Useful, as in our 2D materials z axis is special
        """
        super().__init__(
            nn_strategy=nn_strategy,
            atom_converter=atom_converter,
            bond_converter=bond_converter,
            cutoff=cutoff)

        self.atom_features = atom_features
        if atom_features == "werespecies":
            self.nfeat_node = 2
        elif atom_features == "Z":
            self.nfeat_node = 1
        elif atom_features == "embed":
            self.nfeat_node = None
        else:
            raise ValueError("Invalid atom_features")
        
        self.add_bond_z_coord = add_bond_z_coord
        if add_bond_z_coord:
            self.nfeat_edge = 2
        else:
            self.nfeat_edge = 1

    def get_atom_features(self, structure: Structure):
        if self.atom_features == "werespecies":
            return [[0 if isinstance(i.specie, DummySpecies) else i.specie.Z,
                     i.properties['was']] for i in structure.sites]
        elif self.atom_features == "Z":
            return [[0 if isinstance(i, DummySpecies) else i.Z]
                    for i in structure.species]
        elif self.atom_features == "embed":
            return [0 if isinstance(i, DummySpecies) else i.Z
                    for i in structure.species]

    def convert(self, structure: Structure) -> Dict:
        """
        Convert structure into graph
        Args:
            structure (Structure): pymatgen Structure
        Returns: graph dictionary
        """
        graph = super().convert(structure)
        if self.add_bond_z_coord:
            return self._add_bond_z_coord(graph, structure.cart_coords[:, 2])
        else:
            return graph

    @staticmethod
    def _add_bond_z_coord(graph, z_coords) -> Dict:
        """
        For 2D materials in our project, z-axis is a special axis, as it is orthogonal
        to the material surface. Thus, we include the information on delta z to 
        the edge features.
        """
        new_graph = deepcopy(graph)
        new_graph["bond"] = np.empty((graph["bond"].shape[0], 2), graph["bond"].dtype)
        new_graph["bond"][:, 0] = graph["bond"]
        for k, (i, j) in enumerate(zip(graph["index1"], graph["index2"])):
            new_graph["bond"][k, 1] = abs(z_coords[i] - z_coords[j])
        return new_graph
