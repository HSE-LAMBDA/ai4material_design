#!/bin/bash
python scripts/compute_matminer_features.py --input-name vacancy_pairs/hBN --n-proc=36 --output-folder /output
python scripts/compute_matminer_features.py --input-name low_density_defects/WSe2 --n-proc=36 --output-folder /output
