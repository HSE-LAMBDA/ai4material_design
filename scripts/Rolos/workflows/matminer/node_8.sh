#!/bin/bash
cd ai4material_design
python scripts/compute_matminer_features.py --input-name vacancy_pairs/hBN --n-proc=36 --output-folder /output
