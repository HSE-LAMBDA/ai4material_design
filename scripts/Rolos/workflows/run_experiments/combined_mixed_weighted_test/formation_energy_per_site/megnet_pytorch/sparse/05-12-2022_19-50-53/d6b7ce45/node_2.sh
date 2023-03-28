#!/bin/bash
cd ai4material_design
export WANDB_API_KEY=ae457f48d5eb86299f2fe9c18497a281b029a295
parallel -j 4 python run_experiments.py --n-jobs 2 --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets formation_energy_per_site --experiments combined_mixed_weighted_test --trials ::: stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/5 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/6 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/7 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/8
