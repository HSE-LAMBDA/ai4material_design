import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from functools import cached_property
from pymatgen.core.sites import PeriodicSite

import sys

sys.path.append('.')


class Shells:
    def __init__(self, structure):
        self.structure = structure
        self.nn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)

    @cached_property
    def _all_nn_info(self):
        return self.nn.get_all_nn_info(self.structure)

    def get_nn_shell_info(self, site_idx, shell_idx):
        sites = self.nn._get_nn_shell_info(self.structure, self._all_nn_info, site_idx, shell_idx)
        output = []
        for info in sites:
            orig_site = self.structure[info["site_index"]]
            info["site"] = PeriodicSite(
                orig_site.species,
                np.add(orig_site.frac_coords, info["image"]),
                self.structure.lattice,
                properties=orig_site.properties,
            )
            output.append(info['site'])
        return output


class EOS:
    def __init__(self, num_shells=7) -> None:
        self.num_shells = num_shells

    @staticmethod
    def remove_other_species(structure, center):
        _struct = structure.copy()
        # _struct.remove_species(set(_struct.species).difference({center.spiece}))
        if (num_unique_species := len(set(structure.species))) == 1:
            return Structure.from_sites(
                [site for site in structure if site.properties['center_index'] is not None] + [center])
        elif num_unique_species == 2:
            sites = [site for site in structure if site.specie != center.specie]
            sites.append(center)
            return Structure.from_sites(sites)
        else:
            raise NotImplementedError

    @staticmethod
    def get_distance_of_atoms_on_z_plane(center, sites):
        # return sorted({np.linalg.norm(site.coords - center.coords).round(3) for site in sites if site.coords[2].round(3) == center.coords[2].round(3)})
        return sorted(
            {d for site in sites if (d := np.linalg.norm(site.coords[..., :2] - center.coords[..., :2]).round(3)) != 0})

    def get_shell(self, structure, site_idx, num_shells):
        shells = []
        # for i in range(1, num_shells):
        # shells.append(
        return self.shells_obj.get_nn_shell_info(site_idx, num_shells)
        # return shells

    def add_site_index_to_structure(self, structure):
        for i, site in enumerate(structure):
            site.properties['site_index'] = i
        return structure

    def find_center_index(self, structure, index):
        for i, site in enumerate(structure):
            if site.properties.get('center_index', -1) == index:
                return i

    def get_augmented_struct(self, structure):
        for center_idx, site in enumerate(structure):
            # add center index
            structure[center_idx].properties['center_index'] = center_idx
            # get shells finder object
            self.shells_obj = Shells(structure)
            # get the shells
            shells_sites = self.get_shell(structure, self.find_center_index(structure, center_idx), self.num_shells) + [
                site]
            # remove all other species
            _struct = self.remove_other_species(Structure.from_sites(shells_sites), site)
            # add shells to the site
            site.properties['shells'] = self.get_distance_of_atoms_on_z_plane(site, [site for site in _struct])
            # delete the center index
            del site.properties['center_index']

        assert all(map(lambda s: 'shells' in s.properties, structure))
        return structure