import os
import numpy as np
from tqdm import tqdm
from pymatgen.io.vasp.outputs import Poscar, Outcar, Oszicar, Wavecar, Xdatcar, Vasprun
from pymatgen.core.structure import Structure


def compute_hl_from_vasprun_file(vasprun_file):
    assert len(vasprun_file.eigenvalues) == 1
    values = list(vasprun_file.eigenvalues.items())[0][1][0]
    densities = values[:, 1]
    energies = values[:, 0]
    ind = np.argmax(densities[:-1] - densities[1:])
    assert energies[ind] < energies[ind+1]
    return energies[ind], energies[ind+1]

def main():
    storage_directory = '/mnt/storage/new_materials/data/ai4material_design/datasets' +\
        '/dichalcogenides8x8_vasp_nus_202110/mos2_8x8_5933'
    vasps = os.listdir(storage_directory)

    energies = []
    structures = []
    energies_fermi = []
    homos = []
    lumos = []
    for vasp in tqdm(vasps):
        energy_output = os.path.join(storage_directory, vasp, '01_relax', 'OSZICAR')
        energy_file = Oszicar(energy_output)
        energies.append(energy_file.final_energy)

        structure_output = os.path.join(storage_directory, vasp, '01_relax', 'POSCAR')
        structure_file = Poscar.from_file(structure_output)
        structures.append(structure_file.structure)
        try:
            vasprun_output = os.path.join(storage_directory, vasp, '01_relax', 'vasprun.xml')
            vasprun_file = Vasprun(vasprun_output)
            energies_fermi.append(vasprun_file.efermi)
            homo, lumo = compute_hl_from_vasprun_file(vasprun_file)
            homos.append(homo)
            lumos.append(lumo)
        except Exception:
            energies_fermi.append(np.NaN)
            homos.append(np.NaN)
            lumos.append(np.NaN)
            print(f"Error while reading {vasp}")
    print(energies[:10])
    print(homos[:10])
    print(structures[-2:])

if __name__ == "__main__":
    main()