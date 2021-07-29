from pymatgen.core.periodic_table import DummySpecies
from megnet.data.crystal import CrystalGraph


class VacancyAwareStructureGraph(CrystalGraph):
    @staticmethod
    def get_atom_features(structure):
        return [0 if isinstance(i.specie, DummySpecies) else i.specie.Z for i in structure.sites]
