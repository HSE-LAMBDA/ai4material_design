#!/bin/bash

#SBATCH --job-name ai4mat
#SBATCH --output experiment-%J.log
#SBATCH --export=ALL

module load Python/Anaconda_v10.2019


source /home/${USER}/.bashrc
source activate exp1 

time mpirun python run_experiments.py --experiments MoS2-plain-cv --trials gemnet-full --wandb-entity ${WANDB_ENTITY} --gpus 0 