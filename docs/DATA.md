# Dataset
We provide the data in two formats: [processed](#processed-data) with only the variables used in our paper for ML model evaluation and the full [raw VASP output](#raw-vasp-output).
# Processed data
The processed data contain the unrelaxed structures, energies, formation energies, HOMO, LUMO and derived variables.

## Downloading
The archive can be downloaded and viewed directly at [the Constructor Research Platform](https://research.constructor.tech/open/2d-materials-point-defects).

Alternatively, the data are available in DVC:
1. Clone the repository
2. Ensure that [DVC[S3]](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3) is installed, for example by running `pip install dvc[s3]`
3. Download the datasets ```dvc pull -R processed-high-density processed-low-density datasets/processed/{high,low}_density_defects datasets/csv_cif/high_density_defects/{MoS2,WSe2,BP_spin,GaSe_spin,InSe_spin,hBN_spin}_500 datasets/csv_cif/low_density_defects/{MoS2,WSe2}```

## File format
### `defects.csv.gz`
1. `_id` unique structure identifier
2. `descriptor_id` identifier of the defect type as specified in `descriptors.csv`
3. `defect_id` unused
4. `energy` total potential energy of the system as reported by VASP, eV
5. `energy_per_atom` total potential energy of the system divided by the number of atoms, eV
6. `fermi_level` Fermi level, eV
7. `homo` is highest occupied molecular orbital (HOMO) energy, eV
8. `lumo` is lowest unoccupied molecular orbital (LUMO) energy, eV
9. `normalized_homo` is HOMO value normalised respective to the host valence band maximum (VBM) (see section ["DFT computations" in the paper](https://www.nature.com/articles/s41699-023-00369-1#Sec8)), eV
10. `normalized_homo` is LUMO value normalised respective to the host valence band maximum (VBM) (see section ["DFT computations" in the paper](https://www.nature.com/articles/s41699-023-00369-1#Sec8)), eV
11. `E_1` is the energy of the first Kohn–Sham orbital of the structure with defect (see section ["DFT computations" in the paper](https://www.nature.com/articles/s41699-023-00369-1#Sec8)), eV
12. `homo_lumo_gap` is the band gap, LUMO - HOMO, eV
13. `total_mag` is the total magnetisation
14. `*_{majority,minority}` are the corresponding quantities computed for the majority and minority spin channels for materials computed with spin
15. `band_gap` [OBSOLETE](https://github.com/materialsproject/pymatgen/issues/455)
### `targets.csv.gz`
Same as `defects.csv.gz` plus additional derivative variables:
1. `formation_energy` is the defect formation energy, computed according [equation 1 from the paper](https://www.nature.com/articles/s41524-023-01062-z#Sec5)
2. `formation_energy_per_site` is the defect formation energy divided by the number of defects according to [equation 2 from the paper](https://www.nature.com/articles/s41524-023-01062-z#Sec5)
3. `*_{min,max}` are the minimim and maximum of quantities with the respect to to different spin channels
### `initial.tar.gz`
The archive `initial.tar.gz` contains the unrelaxed structures in the [CIF format](https://doi.org/10.1107%2FS010876739101067X). Names correspond to the unique identifiers `_id` in `defects.csv.gz`. Note that the structures were relaxed prior to computing the properties.
### `descriptors.csv`
1. `_id` unique identifier of the defect type, corresponds to the `descriptor_id` column in `defects.csv`
2. `description` is a short semantic abbreviation of the defect type
3. `base` is the chemical formula of the pristine material
4. `cell` is the supercell size
5. `defects` is a dictionary describing each point defect
### `elements.csv`
Contains chemical potentials (in eV) of the elements, to be used in formation energy computation.
### `initial_structures.csv`
Contains the properties of pristine material.
1. `base` is the chemical formula of the pristine material
2. `cell_size` is the supercell size
3. `energy` total potential energy of the system, eV
4. `fermi` is the Fermi level, eV
5. `E_1` is the energy of the first Kohn–Sham orbital of the pristine structure (see section ["DFT computations" in the paper](https://www.nature.com/articles/s41699-023-00369-1#Sec8)), eV
6. `E_VBM` is the energy of the valence band maximum of pristine structure
### `unit_cells/*.cif`
Unit cells of the pristine materials used to produce the structures in the folder.

# Raw VASP output
The raw VASP output, including the relaxation trajectories, is available in DVC:
1. Clone the repository
2. Ensure that [DVC[S3]](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3) is installed, for example by running `pip install dvc[s3]`
3. Download the VASP output: `dvc pull -R datasets/raw_vasp/high_density_defects datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110`
4. Some of the data are packed into `tar.gz`, as its unpacked size is ~300Gb. You might want to use [ratarmount](https://github.com/mxmlnkn/ratarmount) to work with it.