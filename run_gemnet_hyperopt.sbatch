#!/bin/bash

#SBATCH --job-name ai4mat   ## name that will show up in the queue
#SBATCH --output gemnet_log/%x_%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=4  ## the number of threads allocated to each task
#SBATCH --gpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --time=7-00:01:00  ## time for analysis (day-hour:min:sec)
## SBATCH --mail-user al-maeeni@hse.ru  ## your email address
#SBATCH --export=ALL
## SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
## SBATCH --mail-type END  ## slurm will email you when your job ends
## SBATCH --mail-type FAIL  ## slurm will email you when your job fails
## Load modules
#SBATCH --array=1-50 ## the number of trials to run


module load Python/Anaconda_v11.2020
eval "$(conda shell.bash hook)"
conda activate ai4mat


trial_path=$(ls trials/gemnet/16-11-2022_20-05-04/*.yaml | sed -n ${SLURM_ARRAY_TASK_ID}p)
trial=$(basename $trial_path .yaml)
echo "Processing trial: ${trial_path}"
srun --ntasks=1 --gpus=1 --exclusive -N1 -n1 --cpus-per-task=4 python run_experiments.py --experiments combined_mixed_upsampling_minority_validation  --trials gemnet/16-11-2022_20-05-04/${trial} --wandb-entity abdalaziz-rashid --gpus 0 &
wait
