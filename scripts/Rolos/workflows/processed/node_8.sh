#!/bin/bash
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
python scripts/parse_csv_cif.py --input-name=high_density_defects/MoS2_500 --normalize-homo-lumo --fill-missing-band-properties --output-folder /output/
fi
