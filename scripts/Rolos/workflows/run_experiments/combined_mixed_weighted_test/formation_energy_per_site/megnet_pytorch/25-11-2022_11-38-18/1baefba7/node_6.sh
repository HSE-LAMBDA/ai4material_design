#!/bin/bash
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
source scripts/Rolos/wandb_config.sh
python run_experiments.py --n-jobs 8 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets formation_energy_per_site --experiments combined_mixed_weighted_test --trials stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7/6
fi
