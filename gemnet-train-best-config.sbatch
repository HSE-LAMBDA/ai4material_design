#!/bin/bash

#SBATCH --job-name ai4mat   ## name that will show up in the queue
#SBATCH --output gemnet_log/debug_%x_%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=2  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=4  ## the number of threads allocated to each task
#SBATCH --gpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --time=7-00:01:00  ## time for analysis (day-hour:min:sec)
## SBATCH --mail-user al-maeeni@hse.ru  ## your email address
#SBATCH --export=ALL
## SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
## SBATCH --mail-type END  ## slurm will email you when your job ends
## SBATCH --mail-type FAIL  ## slurm will email you when your job fails
## Load modules


module load Python/Anaconda_v11.2020
eval "$(conda shell.bash hook)"
conda activate latest-ai4mat


formation_energy_per_site_trial='b5723f85'
homo_lumo_gap_min_trial='c366c47e'

echo "Processing trial: ${trial_path}"
# srun --ntasks=1 --gpus=1 --exclusive -N1 -n1 --cpus-per-task=4 python run_experiments.py --experiments combined_mixed_weighted_test --trials gemnet/debug/${formation_energy_per_site_trial}__intensive --wandb-entity abdalaziz-rashid --gpus 0 --targets formation_energy_per_site &
# srun --ntasks=1 --gpus=1 --exclusive -N1 -n1 --cpus-per-task=4 python run_experiments.py --experiments combined_mixed_weighted_test --trials gemnet/debug/${homo_lumo_gap_min_trial}__intensive --wandb-entity abdalaziz-rashid --gpus 0 --targets homo_lumo_gap_min &
srun --ntasks=1 --gpus=1 --exclusive -N1 -n1 --cpus-per-task=4 python run_experiments.py --experiments combined_mixed_weighted_test --trials gemnet/debug/${formation_energy_per_site_trial}__150 --wandb-entity abdalaziz-rashid --gpus 0 --targets formation_energy_per_site &
srun --ntasks=1 --gpus=1 --exclusive -N1 -n1 --cpus-per-task=4 python run_experiments.py --experiments combined_mixed_weighted_test --trials gemnet/debug/${homo_lumo_gap_min_trial}__150 --wandb-entity abdalaziz-rashid --gpus 0 --targets homo_lumo_gap_min &

wait