#!/bin/bash
cd ai4material_design
source scripts/Rolos/wandb_config.sh
parallel -j 2 python run_experiments.py --n-jobs 4 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets homo_lumo_gap_min --experiments combined_mixed_weighted_test --trials ::: stability/catboost/29-11-2022_13-16-01/1b1af67c/3 stability/catboost/29-11-2022_13-16-01/1b1af67c/4
