#!/bin/bash
cd ai4material_design
export WANDB_API_KEY=ae457f48d5eb86299f2fe9c18497a281b029a295
parallel -j 4 python run_experiments.py --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets homo_lumo_gap_min --experiments combined_mixed_weighted_test --trials ::: stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/1 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/2 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/3 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/4
