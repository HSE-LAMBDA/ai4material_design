#!/bin/bash
cd ai4material_design
python scripts/compute_matminer_features.py --input-name low_density_defects/WSe2 --n-proc=36 --output-folder /output
