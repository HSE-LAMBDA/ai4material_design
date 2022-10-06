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
            return ValueError(
                "Input array must be 2-dimensional. Use plain GaussianDistance."
            )
        return np.concatenate(
            [super(FlattenGaussianDistance, self).convert(arr) for arr in d.T], axis=1
        )


class VacancyAwareStructureGraph(CrystalGraph):
    def __init__(
        self,
        nn_strategy: Union[str, NearNeighbors] = "MinimumDistanceNNAll",
        atom_converter: Converter = None,
        bond_converter: Converter = None,
        cutoff: float = 5.0,
        atom_features: str = "embed",
        add_bond_z_coord: bool = True,
        add_eos_indices: bool = False,
        add_eos_parity: bool = False,
    ):
        """ "
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
            cutoff=cutoff,
        )

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
        self.add_eos_indices = add_eos_indices
        self.add_eos_parity = add_eos_parity
        if add_eos_indices:
            if add_eos_parity:
                self.nfeat_edge = 4
            else:
                self.nfeat_edge = 3
        elif add_bond_z_coord:
            self.nfeat_edge = 2
        else:
            self.nfeat_edge = 1

        self.shell_r = {
            # Molybdenum
            42: [
                3.1903157,
                5.52578888,
                6.3806314,
                8.44078195,
                9.5709471,
                11.05157777,
                12.7612628,
                13.90626365,
                14.6198631,
                15.95157841,
                16.57736658,
                17.76292598,
                19.14189411,
                19.92351508,
                20.92029899,
                22.10315554,
                23.0056936,
                24.08635533,
                24.91716209,
                26.11385943,
                27.25806929,
                28.35614618,
                29.23972634,
                30.43367196,
                31.42096565,
                33.15473323,
                34.5085405,
                35.0934727,
                36.79251543,
                37.6132673,
                38.2837884,
                39.84703024,
                40.7312241,
                41.47410401,
                42.92130802,
                43.85958953,
                46.01138737,
                46.99628483,
                49.11428583,
                50.64469161,
                52.22771883,
                54.2353669,
                55.25788885,
                57.4256826,
                59.08547354,
                59.60001573,
                60.61599822,
                63.56658999,
                71.05170424,
                71.83525541,
                74.20485028,
                75.96703751,
                77.3610443,
                81.39995314,
                84.40781938,
                85.66458061,
                87.42862117,
                98.89978668,
                101.94044664,
                109.73016521,
                112.74956631,
                121.19001135,
                124.21763646,
                221.03155538,
                223.80013472,
                226.55734427,
                229.32578662,
                232.08313315,
                234.8514449,
            ],
            # Sulfur
            16: [
                3.12976873,
                4.46917963,
                5.52578888,
                6.35057439,
                7.10689168,
                8.44078195,
                9.00234704,
                9.5709471,
                11.05157777,
                12.7612628,
                13.90626365,
                15.95157841,
                17.76292598,
                19.14189411,
                20.92029899,
                22.10315554,
                24.08635533,
                24.91716209,
                27.25806929,
                30.43367196,
                33.15473323,
                34.5085405,
                37.6132673,
                38.2837884,
                39.84703024,
                40.7312241,
                41.47410401,
                42.92130802,
                43.85958953,
                46.01138737,
                46.99628483,
                49.11428583,
                52.22771883,
                55.25788885,
                57.4256826,
                60.61599822,
                63.56658999,
                71.83525541,
                74.20485028,
                77.3610443,
                85.66458061,
                87.42862117,
                98.89978668,
                109.73016521,
                121.19001135,
                124.21763646,
                221.03155538,
                223.80013472,
                226.55734427,
                229.32578662,
                232.08313315,
                234.8514449,
            ],
        }

    def get_atom_features(self, structure: Structure):
        if self.atom_features == "werespecies":
            return [
                [
                    0 if isinstance(i.specie, DummySpecies) else i.specie.Z,
                    i.properties["was"],
                ]
                for i in structure.sites
            ]
        elif self.atom_features == "Z":
            return [
                [0 if isinstance(i, DummySpecies) else i.Z] for i in structure.species
            ]
        elif self.atom_features == "embed":
            return [
                0 if isinstance(i, DummySpecies) else i.Z for i in structure.species
            ]

    def convert(self, structure: Structure) -> Dict:
        """
        Convert structure into graph
        Args:
            structure (Structure): pymatgen Structure
        Returns: graph dictionary
        """
        graph = super().convert(structure)
        if self.add_eos_indices: # this add z coords by default
            return self.add_eos_idx(graph, structure)
        elif self.add_bond_z_coord:
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
    
    def add_eos_idx(self, graph, struct) -> Dict:
        # raise if z coords are not added or handle
        new_graph = deepcopy(graph)
        # Dimensions depend on self.add_eos_parity
        if self.add_eos_parity:
            dim = 4
        else:
            dim = 3
        new_graph["bond"] = np.empty((graph["bond"].shape[0], dim), graph["bond"].dtype)
        new_graph["bond"][:, 0] = graph["bond"]

        for k, (i, j) in enumerate(zip(graph["index1"], graph["index2"])):
            # add z coords
            new_graph["bond"][k, 1] = abs(struct.cart_coords[:, 2][i] - struct.cart_coords[:, 2][j])
            if new_graph["atom"][i] == new_graph["atom"][j]:
                # argsearch_sorted_lower
                # Let's start indexing from 1
                # raddi=[1.1, 2.3, 3, 4]
                # bond_length=2.5
                # 2
                # [1.1, 2.3, 3, 5], 4
                # 3
                # [1, 2], 0.5
                # 0
                # argsearch_sorted_nearest
                # [1.1, 2.3, 3.1] 1.100121312
                # 0
                # [1.1, 2.3, 3.1] 2.2100121312
                # 1
                new_graph["bond"][k, 2] = np.searchsorted(self.shell_r[new_graph["atom"][i][1]], new_graph["bond"][k, 0])
            else:
                new_graph["bond"][k, 2] = np.isclose(
                        self.shell_r[new_graph["atom"][i][1]],
                        new_graph["bond"][k, 0],
                        rtol=1,
                        atol=1e-01,
                    ).argmax()
            if self.add_eos_parity:
                new_graph["bond"][k, 3] = new_graph["bond"][k, 2] % 2
        return new_graph
