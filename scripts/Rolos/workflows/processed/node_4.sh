#!/bin/bash
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
python scripts/parse_csv_cif.py --input-name=low_density_defects/WSe2 --fill-missing-band-properties --normalize-homo-lumo --output-folder /output/
fi
