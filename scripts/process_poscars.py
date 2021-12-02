import os
import numpy as np
from tqdm import tqdm
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core.structure import Structure


def extract_data_from_vasp(vasp: str) -> dict:
    data = {}
    try:
        vasprun_output = os.path.join(storage_directory, vasp, '01_relax', 'vasprun.xml')
        vasprun_file = Vasprun(vasprun_output)
    except Exception as e:
        print(e, vasp, "Error in vasp reading")
    else:
        data['energy'] = vasprun_file.final_energy
        data['final_structure'] = vasprun_file.final_structure
        data['fermi_level'] = vasprun_file.efermi
        _, data['homo'], data['lumo'], _ = vasprun_file.eigenvalue_band_properties
        data['band_gap'] = vasprun_file.get_band_structure().get_band_gap()['energy']
    return data


def main():
    storage_directory = '/mnt/storage/new_materials/data/ai4material_design/datasets' +\
        '/dichalcogenides8x8_vasp_nus_202110/mos2_8x8_5933'
    vasps = os.listdir(storage_directory)
    
    properties = list(map(extract_data_from_vasp, tqdm(vasps))
    print(properties[0])

if __name__ == "__main__":
    main()