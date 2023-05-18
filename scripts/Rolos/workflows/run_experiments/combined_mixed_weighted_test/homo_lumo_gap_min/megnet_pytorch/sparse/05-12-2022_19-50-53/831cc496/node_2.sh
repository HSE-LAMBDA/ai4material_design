#!/bin/bash
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
source scripts/Rolos/wandb_config.sh
parallel -j 4 python run_experiments.py --n-jobs 2 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets homo_lumo_gap_min --experiments combined_mixed_weighted_test --trials ::: stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/5 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/6 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/7 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/8
fi
