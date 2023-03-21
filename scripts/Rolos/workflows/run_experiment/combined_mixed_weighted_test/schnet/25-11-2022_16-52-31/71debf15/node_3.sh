#!/bin/bash
cd ai4material_design
export WANDB_API_KEY=ae457f48d5eb86299f2fe9c18497a281b029a295
parallel -j 1 python run_experiments.py --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets formation_energy_per_site --experiments combined_mixed_weighted_test --trials ::: stability/schnet/25-11-2022_16-52-31/71debf15/3
