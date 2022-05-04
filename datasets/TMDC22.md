The data contains MoS2 structures simulated with VASP. Each strucure is relaxed, and then the relevant properties are computed.

Our calculations are based on density functional theory (DFT) using the PBE functional as implemented in the Vienna Ab Initio Simulation Package (VASP). The interaction between the valence electrons and ionic cores is described within the projector augmented (PAW) approach with a plane‐wave energy cutoff of 500 eV. Spin polarization was included for all the calculations. The monolayer of MoS2 and defects calculations were performed using an 8x8
supercell, and the Brillouin zone was sampled using a (3x3x1) Monkhorst‐Pack grid. We use periodic boundary conditions, and add a 15Å vacuum space above the material surface to avoid interaction between neighboring layers. In the structural energy minimization, the atomic coordinates are allowed to relax until the forces on all the atoms are less than 0.01 eV/Å. The energy tolerance is 10^(-6) eV. 

We compute the formation energy, i.e., the energy required to create a defect as

<img src="https://render.githubusercontent.com/render/math?math=E_{f} = E_{D}-E_{\text{MoS}_2}+\sum_{i\in\{\text{Mo}, \text{S}\}}{n_i \mu_i}-\sum_{i\in\{\text{W}, \text{Se}\}}{m_i \mu_i}">
          
where $E_{D}$ is the total energy of the structure with defects, $E_{\text{MoS}_2}$ is the total energy of the pristine \ce{MoS_2}, $n_i$ is the number of atoms transferred from the supercell to a chemical reservoir, $m_i$ is the number of atoms transferred from a chemical reservoir to the supercell to form the substitution-type defects, and $\mu_i$ is the chemical potential of $i$-th element. Finally, to make the results better comparable across examples with different numbers of defects, we normalize the formation energy by dividing it by the number of defect sites:

<img src="https://render.githubusercontent.com/render/math?math=E'_{f} = E_f/N_d,">

where $N_d$ is the number of defects in the structure.

The highest occupied molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO) energies are computed respective to the host valence band maximum (VBM) and are normalized according to 

<img src="https://render.githubusercontent.com/render/math?math=E_\text{HOMO} = E_\text{HOMO}^D-E_1^D-(E_\text{VBM}^\text{pristine}-E_1^\text{pristine})">

Where $E_\text{HOMO}^D$ is the eigenvalue of the highest occupied Kohn-Sham states of defects, $E_\text{VBM}^\text{pristine}$ is the eigenvalue of the valence band maximum of pristine \ce{MoS_2}, $E_1^D$ and $E_1^\text{pristine}$ are the energy of the lowest Kohn-Sham orbital of the calculated defect and pristine \ce{MoS_2} structures. Bangap is computed as the difference between LUMO and HOMO.


## File format
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
