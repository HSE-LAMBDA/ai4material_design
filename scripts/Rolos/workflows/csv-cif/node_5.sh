#!/bin/bash
cd ai4material_design
python scripts/vasp_to_csv_cif.py --input-vasp "datasets/raw_vasp/high_density_defects/WSe2_500" --input-structures-list "datasets/POSCARs/WSe2" --POSCARs-in-input-list --output-csv-cif "/output/datasets/csv_cif/high_density_defects/WSe2_500" --pristine-folder datasets/others/pristine_high_density
