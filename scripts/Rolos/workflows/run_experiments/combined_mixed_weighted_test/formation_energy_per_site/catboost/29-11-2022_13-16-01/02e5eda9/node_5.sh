#!/bin/bash
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
source scripts/Rolos/wandb_config.sh
parallel -j 2 python run_experiments.py --n-jobs 4 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets formation_energy_per_site --experiments combined_mixed_weighted_test --trials ::: stability/catboost/29-11-2022_13-16-01/02e5eda9/9 stability/catboost/29-11-2022_13-16-01/02e5eda9/10
fi
