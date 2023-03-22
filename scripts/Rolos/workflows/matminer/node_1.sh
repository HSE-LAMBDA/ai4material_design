#!/bin/bash
cd ai4material_design
python scripts/compute_matminer_features.py --input-name vacancy_pairs/InSe --n-proc=36 --output-folder /output
python scripts/compute_matminer_features.py --input-name high_density_defects/MoS2_500 --n-proc=36 --output-folder /output
