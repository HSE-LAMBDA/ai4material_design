#!/bin/bash

#SBATCH --job-name ai4mat   ## name that will show up in the queue
#SBATCH --output megnet_full.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=4  ## the number of threads allocated to each task
#SBATCH --gpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --time=7-00:01:00  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-user ipromanov@edu.hse.ru  ## your email address
#SBATCH --export=ALL

## SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
#SBATCH --mail-type END  ## slurm will email you when your job ends
#SBATCH --mail-type FAIL  ## slurm will email you when your job fails
## Load modules

module load Python/Anaconda_v11.2020
eval "$(conda shell.bash hook)"
conda activate common

srun --ntasks=1 --gpus=1 --exclusive -N1 -n1 --cpus-per-task=1 python run_experiments.py --experiments combined_mixed_weighted_test  --trials 1baefba7 --wandb-entity hse_lambda --gpus 0 &
wait
