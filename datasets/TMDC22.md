The data contains MoS2 structures simulated with VASP. Each strucure is relaxed, and then the relevant properties are computed.


## Format
### `defects.csv`
1) (unnamed first column) structure number in the file
2) `_id` unique structure identifier
3) `descriptor_id` identifier of the defect type as specified in `descriptors.csv`
4) `defect_id` unused
5) `energy` total potential energy of the system, eV
6) `energy_per_atom` total potential energy of the system divided by the number of atoms, eV
7) `fermi_level` Fermi level, eV
8) `homo` highest occupied molecular orbital (HOMO) energy, eV
9) `lumo` lowest unoccupied molecular orbital (LUMO) energy, eV
10) `normalized_homo` is HOMO value normalised respective to the host valence band maximum (VBM) (see section "DFT computations" in the paper), eV
11) `normalized_homo` is LUMO value normalised respective to the host valence band maximum (VBM) (see section "DFT computations" in the paper), eV
12) `band_gap` is the band gap, LUMO - HOMO, eV
### `initial`
The folder `initial` contains the unrelaxed structures in the [CIF format](https://doi.org/10.1107%2FS010876739101067X). Names correspond to the unique identifiers `_id` in `defects.csv`. Note that the structures were relaxed prior to computing the properties.
### `descriptors.csv`
1) `_id` unique identifier of the defect type, corresponds to the `descriptor_id` column in `defects.csv`
2) `description` is a short semantic abbreviation of the defect type
3) `base` is the chemical formula of the pristine material
4) `cell` is the supercell size
5) `pbc` is WTF, in DFT pbc were in all dimensions
6) `defects` is a dictionary describing each point defect
### `elements.csv`
Contains chemical potentials (in eV) of the elements, to be used in formation energy computation.
### `initial_structures.csv`
Contains the properties of pristine material.
1) (unnamed first column) structure number in the file
2) `base` is the chemical formula of the pristine material
3) `cell_length` is the supercell length, the supercell size is `[cell_length, cell_length, 1]`
4) `energy` total potential energy of the system, eV
5) `fermi` is the Fermi level, eV
