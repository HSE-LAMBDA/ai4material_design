#!/bin/bash
python scripts/compute_matminer_features.py --input-name vacancy_pairs/InSe --n-proc=36 --output-folder /output
python scripts/compute_matminer_features.py --input-name high_density_defects/InSe_spin_500 --n-proc=36 --output-folder /output
