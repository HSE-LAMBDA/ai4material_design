#!/bin/bash
python run_experiments.py --trials megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 --experiments combined_mixed_all_train --gpu --n-jobs 4 --wandb-entity hse_lambda --save-checkpoints