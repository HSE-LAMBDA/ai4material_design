#!/bin/bash
cd ai4material_design
python scripts/separate_csv_cif.py --input-folder datasets/csv_cif/dichalcogenides_x1s6_202109 --output-folder /output/datasets/csv_cif/low_density_defects_Innopolis-v1/WSe2 --base-material WSe2 --supercell-size 8
