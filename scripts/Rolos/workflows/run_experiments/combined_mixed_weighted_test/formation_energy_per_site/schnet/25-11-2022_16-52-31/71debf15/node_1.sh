#!/bin/bash
cd ai4material_design
source scripts/Rolos/wandb_config.sh
python run_experiments.py --n-jobs 8 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets formation_energy_per_site --experiments combined_mixed_weighted_test --trials stability/schnet/25-11-2022_16-52-31/71debf15/1
