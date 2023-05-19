# Reproducing "Sparse representation for machine learning the properties of defects in 2D materials"
## Note: the data are already here
All input, intermediate, and output data are already available in the repository, you can selectively reproduce the parts you want.
## Setting up the environment
See [ENVIRONMENT.md](./ENVIRONMENT.md)
## Data preprocessing
### VASP -> csv/cif -> pickle
```bash
dvc pull -R datasets/POSCARs datasets/raw_vasp/high_density_defects datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110 datasets/csv_cif/low_density_defects_Innopolis-v1/{MoS2,WSe2}
parallel --delay 3 -j6 dvc repro processed-high-density@{} ::: hBN_spin GaSe_spin BP_spin InSe_spin MoS2 WSe2
parallel --delay 3 -j2 dvc repro processed-low-density@{} ::: MoS2 WSe2
```
Note that unlike GNU Make DVC [currently](https://github.com/iterative/dvc/issues/755) doesn't internally parallelize execution, so we use GNU parallel. We also use `--delay 3` to avoid [DVC lock race](https://github.com/iterative/dvc/issues/755). Computing matmier features can easily take several days, you might want to parallelize it according to your computing setup.
### Matminer
Assuming the resources are available, the step takes around 3 days, you can skip it if don't plan on running CatBoost. 
```bash
dvc repro matminer
```
## Hyperparameter optimisation
### Get the data
```
dvc pull -R processed-high-density processed-low-density datasets/processed/{high,low}_density_defects datasets/experiments/combined_mixed_weighted_test.dvc datasets/experiments/combined_mixed_weighted_validation.dvc
``` 
### Generate the trials
```
python scripts/generate_trials_for_tuning.py --model-name megnet_pytorch --mode random --n-steps 50
python scripts/generate_trials_for_tuning.py --model-name megnet_pytorch/sparse --mode random --n-steps 50
python scripts/generate_trials_for_tuning.py --model-name catboost --mode random --n-steps 50
python scripts/generate_trials_for_tuning.py --model-name gemnet --mode random --n-steps 50
python scripts/generate_trials_for_tuning.py --model-name schnet --mode random --n-steps 50
```
This will create `trials/<model_name>/<date>` folders with trials. Alternatively, you can pull our trials with `dvc pull -R trials`.
### Run experiments on train/validation split
The next step is running those trials combined with `combined_mixed_weighted_validation` experiment according to your compute environmet. For example, on a single GPU:
```bash
python run_experiments.py --experiments combined_mixed_weighted_validation --trials trials/megnet_pytorch/sparse/05-12-2022_19-34-37/0ff69f1c --gpus 0
```
or on CPU:
```bash
python run_experiments.py --experiments combined_mixed_weighted_validation --trials trials/megnet_pytorch/sparse/05-12-2022_19-34-37/0ff69f1c --cpu
```
For running trials for a specific model on a single node, we have a script:
```bash
python scripts/hyperparam_tuning.py --model-name megnet_pytorch --experiment combined_mixed_weighted_validation --wandb-entity hse_lambda --trials-folder trials/megnet_pytorch/sparse/05-12-2022_19-34-37/
```
There is also script `scripts/ASPIRE-1/run_grid_search.sh`, for running on an HPC, but it is specific to our cluster.
### Find the best trials
Use `find_best_trial.py` for every model, e.g.:
```bash
python scripts/find_best_trial.py --experiment combined_mixed_weighted_validation --trials-folder megnet_pytorch/sparse/05-12-2022_19-50-53
```
## Aggregate experiments on train/test split
Since some models (thankfully, not ours) exhibit instability, we repeat training several times for each model - with the same parameters and training data. To fit this into the infrastrucrure we copy the trials. This step was only done on ASPIRE-1, so it would requre some modifications to run on a different cluster (e. g. replace `qsub` with `sbatch`). Note that CatBoost by default is deterministic, so you need to change the random seed manually in the copies of the trials.
```bash
cd scripts/ASPIRE-1
xargs -a stability_trials.txt -L1 ./run_stability_trials.sh 
```
Format of `stability_trials.txt`:

```bash
megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 formation_energy_per_site 12 4 combined_mixed_weighted_test
trial target total_repeats parallel_runs_per_GPU experiment
```
## Quantum oscillations aka E(distance) predictions for MoS2:
```bash
xargs -a MoS2_V2_E.txt -L1 ./run_stability_trials.sh 
```
## Ablation study
Manually prepare the model configurations (aka trials) in `trials/megnet_pytorch/ablation_study`. Put them into a `.txt` and run the experiments:
```bash
cd scripts/ASPIRE-1
xargs ablation_stability.txt -L1 ./run_stability_trials.sh
```
## Result analysis
### LaTeX Tables
If you generated your own trials, you need to replace the trial names. Main results:
```bash
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets formation_energy_per_site --stability-trials stability/schnet/25-11-2022_16-52-31/71debf15 stability/catboost/29-11-2022_13-16-01/02e5eda9 stability/gemnet/16-11-2022_20-05-04/b5723f85 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 --separate-by target --column-format-re stability\/\(?P\<name\>.+\)\/.+/\.+ --paper-results --multiple 1000
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets homo_lumo_gap_min --stability-trials stability/schnet/25-11-2022_16-52-31/2a52dbe8 stability/catboost/29-11-2022_13-16-01/1b1af67c stability/gemnet/16-11-2022_20-05-04/c366c47e stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 --separate-by target --column-format-re stability\/\(?P\<name\>.+\)\/.+/\.+ --paper-results --multiple 1000
```
Ablation:
```bash
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets formation_energy_per_site --stability-trials stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 stability/megnet_pytorch/ablation_study/d6b7ce45-sparse stability/megnet_pytorch/ablation_study/d6b7ce45-sparse-z stability/megnet_pytorch/ablation_study/d6b7ce45-sparse-z-were --separate-by target --print-std --paper-ablation-energy --multiple 1000
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets homo_lumo_gap_min --stability-trials stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 stability/megnet_pytorch/ablation_study/831cc496-sparse{,-z,-z-were} --separate-by target --print-std --paper-ablation-homo-lumo --multiple 1000
```
### Notebook
[`ai4material_design/notebooks/Results tables.ipynb`](../notebooks/Results%20tables.ipynb)
## Quantum oscillations aka E(distance) plots
[`ai4material_design/notebooks/MoS2_V2_plot.ipynb`](../notebooks/MoS2_V2_plot.ipynb)