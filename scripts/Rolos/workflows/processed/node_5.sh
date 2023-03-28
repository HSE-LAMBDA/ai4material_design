#!/bin/bash
cd ai4material_design
python scripts/parse_csv_cif.py --input-name=high_density_defects/GaSe_spin_500 --normalize-homo-lumo --fill-missing-band-properties --output-folder /output
