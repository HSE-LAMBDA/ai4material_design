#!/bin/bash
cd ai4material_design
python scripts/parse_csv_cif.py --input-name=low_density_defects/MoS2 --fill-missing-band-properties --normalize-homo-lumo --output-folder /output/
