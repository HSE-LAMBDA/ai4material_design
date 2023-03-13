#!/bin/bash
python scripts/vasp_to_csv_cif.py --input-vasp "datasets/raw_vasp/high_density_defects/hBN_spin_500" --input-structures-list "datasets/POSCARs/hBN" --POSCARs-in-input-list --output-csv-cif "/output/datasets/csv_cif/high_density_defects/hBN_spin_500" --pristine-folder datasets/others/pristine_high_density --separate-spins
