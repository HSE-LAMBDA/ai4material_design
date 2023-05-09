#!/bin/bash
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
pip install "numpy<1.24.0"
python scripts/compute_matminer_features.py --input-name high_density_defects/WSe2_500 --n-proc=24 --output-folder /output/
fi
