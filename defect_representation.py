from pymatgen.core.periodic_table import DummySpecies
from megnet.data.crystal import CrystalGraph


class VacancyAwareStructureGraph(CrystalGraph):
    n_atom_features = 2
    @staticmethod
    def get_atom_features(structure):
        return [[0 if isinstance(i.specie, DummySpecies) else i.specie.Z,
                 i.properties['was']] for i in structure.sites]
