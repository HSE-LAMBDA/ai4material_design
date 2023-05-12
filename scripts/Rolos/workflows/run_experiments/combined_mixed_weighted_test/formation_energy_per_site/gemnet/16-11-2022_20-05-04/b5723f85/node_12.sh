#!/bin/bash
cd ai4material_design
source scripts/Rolos/wandb_config.sh
python run_experiments.py --n-jobs 8 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets formation_energy_per_site --experiments combined_mixed_weighted_test --trials stability/gemnet/16-11-2022_20-05-04/b5723f85/12
