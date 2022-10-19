#!/bin/bash

#SBATCH --job-name ai4mat   ## name that will show up in the queue
#SBATCH --output gemnet.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=6  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=6  ## the number of threads allocated to each task
#SBATCH --gpus-per-task=2  ## the number of threads allocated to each task
#SBATCH --time=7-00:01:00  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-user al-maeeni@hse.ru  ## your email address
#SBATCH --export=ALL

## SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
#SBATCH --mail-type END  ## slurm will email you when your job ends
#SBATCH --mail-type FAIL  ## slurm will email you when your job fails
## Load modules

module load Python/Anaconda_v11.2020
eval "$(conda shell.bash hook)"
conda activate ai4mat

srun --ntasks=1 --gpus=2 --exclusive -N1 -n1 --cpus-per-task=6 python run_experiments.py --experiments high_density/WSe2_500  --trials gemnet-full --wandb-entity abdalaziz-rashid --gpus 0 1 &
srun --ntasks=1 --gpus=2 --exclusive -N1 -n1 --cpus-per-task=6 python run_experiments.py --experiments high_density/BP_spin_500  --trials gemnet-full --wandb-entity abdalaziz-rashid --gpus 0 1 &
srun --ntasks=1 --gpus=2 --exclusive -N1 -n1 --cpus-per-task=6 python run_experiments.py --experiments high_density/GaSe_500  --trials gemnet-full --wandb-entity abdalaziz-rashid --gpus 0 1 &
srun --ntasks=1 --gpus=2 --exclusive -N1 -n1 --cpus-per-task=6 python run_experiments.py --experiments high_density/hBN_spin_500  --trials gemnet-full --wandb-entity abdalaziz-rashid --gpus 0 1 &
srun --ntasks=1 --gpus=2 --exclusive -N1 -n1 --cpus-per-task=6 python run_experiments.py --experiments high_density/InSe_500  --trials gemnet-full --wandb-entity abdalaziz-rashid --gpus 0 1 &
srun --ntasks=1 --gpus=2 --exclusive -N1 -n1 --cpus-per-task=6 python run_experiments.py --experiments high_density/MoS2_500  --trials gemnet-full --wandb-entity abdalaziz-rashid --gpus 0 1 &
wait
