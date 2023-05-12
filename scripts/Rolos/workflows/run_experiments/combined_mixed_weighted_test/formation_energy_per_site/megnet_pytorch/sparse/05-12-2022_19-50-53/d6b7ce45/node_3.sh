#!/bin/bash
cd ai4material_design
source scripts/Rolos/wandb_config.sh
parallel -j 4 python run_experiments.py --n-jobs 2 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets formation_energy_per_site --experiments combined_mixed_weighted_test --trials ::: stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/9 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/10 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/11 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/12
