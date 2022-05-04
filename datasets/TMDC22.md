The data contains MoS2 structures simulated with VASP. Each strucure is relaxed, and then the relevant properties are computed.


## Format
`defects.csv` columns:
1) (unnamed first column) structure number in the file
2) `_id` unique structure identifier
3) `descriptor_id` identifier of the defect type as specified in `descriptors.csv`
4) `defect_id` unused
5) `energy` total potential energy of the system
6) `energy_per_atom` total potential energy of the system divided by the number of atoms
7) `fermi_level` Fermi level
8) `homo` highest occupied molecular orbital (HOMO) energy
9) `lumo` lowest unoccupied molecular orbital (LUMO) energy
10) `normalized_homo` is HOMO value normalised respective to the host valence band maximum (VBM) (see section "DFT computations" in the paper)
11) `normalized_homo` is LUMO value normalised respective to the host valence band maximum (VBM) (see section "DFT computations" in the paper)
12) `band_gap` is the band gap, LUMO - HOMO
