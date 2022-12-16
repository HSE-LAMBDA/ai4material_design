#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=4:00:00
#PBS -q dgx
#PBS -P 11001786
#PBS -j oe
IMAGE=/home/projects/ai/singularity/nvcr.io/nvidia/pytorch:22.04-py3.sif
cd /home/projects/11001786/ai4material_design/
if [ -n "$TARGETS" ]; then
    TARGETS_FLAG="--targets ${TARGETS}"
fi
parallel singularity exec $IMAGE python run_experiments.py --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda ${TARGETS_FLAG} --experiments ${EXPERIMENT} --trials ::: ${TRIALS}
