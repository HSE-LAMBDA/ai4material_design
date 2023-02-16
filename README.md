# AI for material design
- The overall design is documented in the [flowchart](https://miro.com/welcomeonboard/eUdTWFNlaTZOZkc3NUlqd2o0TXB2QUUxRjFWVGxVcGtrWTJ5U01lbFZ1aFZxTFJRcUNyNG5NMjFaZkZ4S3pHRXwzMDc0NDU3MzU5MDMzOTQ0ODgx?invite_link_id=740759716756)
- Some design decisions are outlined in [RFC](https://docs.google.com/document/d/1Cc3772US-E73yQEMFn444OY9og9blKHpuP21sv9Gdxk/edit?usp=sharing)
- Project log is in [Notion](https://www.notion.so/AI-for-material-design-1f8f321d2ac54245a7af410d838929ae)
- Paper in [Overleaf](https://www.overleaf.com/project/61893015795e7b18e7979f53)

## Setting up the environment
1. [Install Poetry](https://python-poetry.org/docs/#installation)
2. Next steps depend on your setup
   - If you don't want to use vritualenv, for example to use system `torch`, run
   ```
   poetry config virtualenvs.create false --local
   ```
   - If you want to use virtualenv, run
   ```
   poetry shell
   ```
3. Install ordinary dependencies
```
poetry install
```
If it fails, try removing `poetry.lock`. We are forced to support multiple Python versions, so it's imposible to have a single lock file.

4. [Install pytorch](https://pytorch.org/) according to your CUDA/virtualenv/conda situatoin
5. [Install pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) according to your CUDA/virtualenv/conda situatoin
6. [Log in to WanDB](https://docs.wandb.ai/ref/cli/wandb-login), or set `WANDB_MODE=disabled`

## Running the pilot NN model
Below we descrbie a lightweight test run.

0. Pull the inputs from DVC
```
dvc pull datasets/csv_cif/pilot datasets/experiments/pilot-plain-cv datasets/processed/pilot
```

1. Preprocess the data to get targets, pickled full and sparse structures
```
python scripts/parse_csv_cif.py --input-name=pilot --fill-missing-band-properties
```
This creates `datasets/processed/pilot/{data.pickle.gzip,targets.csv}`

2. Run the experiments
This step creates predictions in `datasets/predictions/pilot-plain-cv` and run information at [WanDB](https://wandb.ai/hse_lambda/ai4material_design). Make sure you are logged in to WanDB and use WanDB entity you have access to.
- GPU
   Adjust the `--gpus` and `--processes-per-unit` options to your GPU resources
```
python run_experiments.py --experiments pilot-plain-cv --trials megnet_pytorch-sparse-pilot --gpus 0 1 2 3 --processes-per-unit 2 --wandb-entity hse_lambda
```
- CPU
```
python run_experiments.py --experiments pilot-plain-cv --trials megnet_pytorch-sparse-pilot --cpu --processes-per-unit 8 --wandb-entity hse_lambda
```
- slurm
Modify `slurm-job.sh` with the desired argument and export the required enviroment variables then run `./slurm-job.sh`

3. Plot the plots
```
python scripts/plot.py --experiments pilot-plain-cv --trials megnet_pytorch-sparse-pilot
```
This produces plots in `datasets/plots/pilot-plain-cv`

4. If you want to perform random hyperparameters search on pilot do next steps

- change templates/megnet_pytorch/parameters_to_tune.yaml (if model is not megnet you need to create the directory with model name and two template files)

example
```
model_params:
  model:
    train_batch_size: ['int_min_max', 32, 256]
    vertex_aggregation: ['grid', 'sum', 'max']
  optim:
    factor: ['float_min_max', 0.3, 0.9]
```

the first element in each list must be distribution, these three distributions are now available

```
python scripts/generate_trials_for_tuning.py --model-name megnet_pytorch --mode random --n-steps 5
```

- it will produce folder with trials
- if you want then to run them on hpc cluster you can use

```
python scripts/hyperparam_tuning.py --model-name megnet_pytorch --experiment pilot-plain-cv --wandb-entity hse_lambda --trials-folder {folder name from previous step}
```

## Running a pilot CatBoost model
0. Pull the inputs from DVC
```
dvc pull datasets/csv_cif/pilot datasets/experiments/matminer-test
```

1. Prepare the targets and matminer features  
Can be done with one of the two following commands:  
Compute features on the machine (up to several minutes per structure on single core)
```
python scripts/compute_matminer_features.py --input-name=pilot --n-proc 8
```
OR load existing features
```
dvc pull datasets/processed/pilot/matminer.csv.gz
```

Both scenarios produce `datasets/processed/pilot/matminer.csv.gz`

2. Run the experiments
```
python run_experiments.py --experiments matminer-test --trials catboost-pilot --gpus 0 1 2 3 --wandb-entity hse_lambda
```
This creates predictions in `datasets/predcitions/matminer-test`

3. Plot the plots
```
python scripts/plot.py --experiments matminer-test --trials catboost-pilot
```
This produces plots in `datasets/plots/matminer-test`

## Sparse representation for machine learning the properties of defects in 2D materials (paper)
Reproducing the paper requires roughly four stages. Intermidiate artifacts are saved in DVC, therefore stages can be reproduced selectively.
## Data preprocessing: VASP -> csv/cif -> pickle
```
dvc pull -R datasets/POSCARs datasets/raw_vasp/high_density_defects datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110 datasets/csv_cif/low_density_defects_Innopolis-v1
parallel --delay 3 -j6 dvc repro processed-high-density@{} ::: hBN_spin GaSe_spin BP_spin InSe_spin MoS2 WSe2
parallel --delay 3 -j2 dvc repro processed-low-density@{} ::: MoS2 WSe2
```
Note that unlike GNU Make DVC [currently](https://github.com/iterative/dvc/issues/755) doesn't internally parallelize execution, so we use GNU parallel. We also use `--delay 3` to avoid [DVC lock race](https://github.com/iterative/dvc/issues/755). Computing matmier features can easily take several days, you might want to parallelize it according to your computing setup.
```
dvc repro matminer
```
## Hyperparameter optimisation
### Get the data
```
dvc pull -R datasets/csv_cif/{high,low}_density_defects/ datasets/processed/{high,low}_density_defects datasets/experiments/combined_mixed_weighted_test.dvc datasets/experiments/combined_mixed_weighted_validation.dvc
```
### Generate the trials.
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
```
python run_experiments.py --experiments combined_mixed_weighted_validation --trials trials/megnet_pytorch/sparse/05-12-2022_19-34-37/0ff69f1c --gpus 0
```
or on a CPU:
```
python run_experiments.py --experiments combined_mixed_weighted_validation --trials trials/megnet_pytorch/sparse/05-12-2022_19-34-37/0ff69f1c --cpu
```
For running trials for a specific model on a single node, we have a script:
```
python scripts/hyperparam_tuning.py --model-name megnet_pytorch --experiment combined_mixed_weighted_validation --wandb-entity hse_lambda --trials-folder trials/megnet_pytorch/sparse/05-12-2022_19-34-37/
```
There is also script `scripts/ASPIRE-1/run_grid_search.sh`, for running on an HPC, but it is specific to our cluster.
### Find the best trials
Use `find_best_trial.py` for every model, e.g.:
```
python scripts/find_best_trial.py --experiment combined_mixed_weighted_validation --trials-folder megnet_pytorch/sparse/05-12-2022_19-50-53
```
### Run the experiment on train/test split
Since some models (thankfully, not ours) exhibit instability, we repeat training several times for each model - with the same parameters and training data. To fit this into the infrastrucrure we copy the trials. This step was only done on ASPIRE-1, so it would requre some modifications to run on a different cluster (e. g. replace `qsub` with `sbatch`). Note that CatBoost by default is deterministic, so you need to change the random seed manually in the copies of the trials.
```
cd scripts/ASPIRE-1
xargs -a stability_trials.txt -L1 ./run_stability_trials.sh 
```
Format of `stability_trials.txt`:

```
megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 formation_energy_per_site 12 4 combined_mixed_weighted_test
trial target total_repeats parallel_runs_per_GPU experiment
```
To obtain the data for E(distance) plots for MoS2:
```
xargs -a MoS2_V2_E.txt -L1 ./run_stability_trials.sh 
```
### Ablation study
Manually prepare the model configurations (aka trials) in `trials/megnet_pytorch/ablation_study`. Put them into a `.txt` and run the experiments:
```
xargs ablation_stability.txt -L1 ./run_stability_trials.sh 
```
## Result analysis
### Tables
If you generated your own trials, you need to replace the trial names. Main results:
```
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets formation_energy_per_site --stability-trials stability/schnet/25-11-2022_16-52-31/71debf15 stability/catboost/29-11-2022_13-16-01/02e5eda9 stability/gemnet/16-11-2022_20-05-04/b5723f85 stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 --separate-by target --column-format-re stability\/\(?P\<name\>.+\)\/.+/\.+ --paper-results --multiple 1000
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets homo_lumo_gap_min --stability-trials stability/schnet/25-11-2022_16-52-31/2a52dbe8 stability/catboost/29-11-2022_13-16-01/1b1af67c stability/gemnet/16-11-2022_20-05-04/c366c47e stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 --separate-by target --column-format-re stability\/\(?P\<name\>.+\)\/.+/\.+ --paper-results --multiple 1000
```
Ablation:
```
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets formation_energy_per_site --stability-trials stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 stability/megnet_pytorch/ablation_study/d6b7ce45-sparse stability/megnet_pytorch/ablation_study/d6b7ce45-sparse-z stability/megnet_pytorch/ablation_study/d6b7ce45-sparse-z-were --separate-by target --print-std --paper-ablation-energy --multiple 1000
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets homo_lumo_gap_min --stability-trials stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496 stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7 stability/megnet_pytorch/ablation_study/831cc496-sparse{,-z,-z-were} --separate-by target --print-std --paper-ablation-homo-lumo --multiple 1000
```
### E(distance) plots
Run the notebook `notebooks/MoS2_V2_plot.ipynb` replacing the trial names with your own.
## Additional considerations
### `prepare_data_split.py`
Generates data splits aka experiments. There is no need to do this step to run the existing experiments, the splits are available in DVC. Splits are by design shared between people, so don't overwrite them needlessly. Example:
```
python scripts/prepare_data_split.py --datasets=datasets/csv_cif/pilot --experiment-name=pilot-plain-cv --targets band_gap homo formation_energy_per_site
```
This creates the experiment definition in `datasets/experiments/pilot-plain-cv`

It supports generating cross-validation and train/test splits.
## Rolos demo
### Environment
Follow the corresponding section. If something is missing, please help us by adding it to `pyproject.toml`. CatBoost is not available for Python 3.11, but they [plan](https://github.com/catboost/catboost/issues/2213) to ship it before 03.02.2023.
### Get the data
```
dvc pull processed-low-density processed-high-density datasets/experiments/combined_mixed_weighted_test datasets/experiments/MoS2_V2 matminer
dvc pull -R trials
```
The data will need to be added to the Rolos LFS. DVC credentials currently in the repository will be invalidated when we open the code. In the interest of showcasig Rolos, you might want to `dvc pull` everything (~30 Gb) and push it git LFS.
### Run the experiments
The trials in the commands have optimal hyperparameters. Split the trials over multiple invocations of `run_experiments.py` according to your exection environement. IMHO, a workflow for each experiment, and a node for each trial. Table-printing script and plotting notebook expect that Rolos wrokflows are named `MoS2_V2` and `combined_mixed_weighted_test`, but it's trivial to change. In case there are pymatgen errors, you might need to re-run the last data processing step csv/cif -> pickle, follow `dvc.yaml` for that, something like `dvc repro -s --pull processed-low-density processed-high-density`. For debug, you can use trial `megnet_pytorch/sparse/pilot`. To run on a single gpu, add `--gpus 0`, to run on CPU, add `--cpu`. Since we are running in train/test mode, `run_experiments.py` won't parallelize between GPUs.
Aggregate:
```
WANDB_MODE=disabled python run_experiments.py --experiments combined_mixed_weighted_test --targets formation_energy_per_site --output-folder /output --trials schnet/25-11-2022_16-52-31/71debf15 catboost/29-11-2022_13-16-01/02e5eda9 gemnet/16-11-2022_20-05-04/b5723f85 megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 megnet_pytorch/25-11-2022_11-38-18/1baefba7
```
MoS2 E(distance):
```
WANDB_MODE=disabled python run_experiments.py --experiments MoS2_V2 --targets formation_energy_per_site --output-folder /output --trials schnet/25-11-2022_16-52-31/71debf15 catboost/29-11-2022_13-16-01/02e5eda9 gemnet/16-11-2022_20-05-04/b5723f85 megnet_pytorch/sparse/d6b7ce45_no_resample megnet_pytorch/25-11-2022_11-38-18/1baefba7
```

### Print the aggregate table
ASCII
```
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets formation_energy_per_site --trials schnet/25-11-2022_16-52-31/71debf15 catboost/29-11-2022_13-16-01/02e5eda9 gemnet/16-11-2022_20-05-04/b5723f85 megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 megnet_pytorch/25-11-2022_11-38-18/1baefba7 --separate-by target --column-format-re \(?P\<name\>.+\)\/.+/\.+ --storage-root /home/coder/project/rolos_workflow_data/combined_mixed_weighted_test/current/data --multiple 1000
```
LaTeX
```
python scripts/summary_table_lean.py --experiment combined_mixed_weighted_test --targets formation_energy_per_site --trials schnet/25-11-2022_16-52-31/71debf15 catboost/29-11-2022_13-16-01/02e5eda9 gemnet/16-11-2022_20-05-04/b5723f85 megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45 megnet_pytorch/25-11-2022_11-38-18/1baefba7 --separate-by target --column-format-re \(?P\<name\>.+\)\/.+/\.+ --storage-root /tmp/rolos /home/coder/project/rolos_workflow_data/combined_mixed_weighted_test/current/data --multiple 1000 --paper-results
```
### Draw the E(distance) plot
Run the notebook `notebooks/MoS2_V2_plot.ipynb`.