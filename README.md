# ML for point defects in 2D materials
- The overall design is documented in the [flowchart](https://miro.com/welcomeonboard/eUdTWFNlaTZOZkc3NUlqd2o0TXB2QUUxRjFWVGxVcGtrWTJ5U01lbFZ1aFZxTFJRcUNyNG5NMjFaZkZ4S3pHRXwzMDc0NDU3MzU5MDMzOTQ0ODgx?invite_link_id=740759716756)
- Some design decisions are outlined in [RFC](https://docs.google.com/document/d/1Cc3772US-E73yQEMFn444OY9og9blKHpuP21sv9Gdxk/edit?usp=sharing)
- Project log is in [Notion](https://www.notion.so/AI-for-material-design-1f8f321d2ac54245a7af410d838929ae)
- Paper in [Overleaf](https://www.overleaf.com/project/61893015795e7b18e7979f53)

## Setting up the environment
### Locally
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
### Rolos
Should work out-of-the-box
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

# Sparse representation for machine learning the properties of defects in 2D materials (paper)
Intermidiate artifacts are saved in DVC, therefore stages can be reproduced selectively.

## Rolos important note
After running a workflow, you need to grab the outputs from workflow and add them to git:
```bash
export WORKFLOW="<workflow name>"
# Example:
# export WORKFLOW="Combined test MegNet sparse"
cp -r "rolos_workflow_data/${WORKFLOW}/current/data/datasets" ai4material_design/
git add ai4material_design/datasets
git commit -m "Workflow ${WORKFLOW} results"
git push
```
## Data preprocessing: VASP -> csv/cif -> pickle & matminer
### Locally
```bash
dvc pull -R datasets/POSCARs datasets/raw_vasp/high_density_defects datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110 datasets/csv_cif/low_density_defects_Innopolis-v1
parallel --delay 3 -j6 dvc repro processed-high-density@{} ::: hBN_spin GaSe_spin BP_spin InSe_spin MoS2 WSe2
parallel --delay 3 -j2 dvc repro processed-low-density@{} ::: MoS2 WSe2
```
Note that unlike GNU Make DVC [currently](https://github.com/iterative/dvc/issues/755) doesn't internally parallelize execution, so we use GNU parallel. We also use `--delay 3` to avoid [DVC lock race](https://github.com/iterative/dvc/issues/755). Computing matmier features can easily take several days, you might want to parallelize it according to your computing setup.
```bash
dvc repro matminer
```
### Rolos
To reduce the repository size, raw VASP files are not stored in Rolos, you'll need to download them from DVC. Prior to that, you'll need to increase the project size, 100 Gb should be sufficient.
```bash
dvc pull datasets/raw_vasp/high_density_defects/{BP,GaSe,hBN,InSe}_spin*.dvc
dvc pull datasets/raw_vasp/high_density_defects/{MoS,WSe}2_500.dvc
dvc pull datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110/*.tar.gz.dvc

git add datasets/raw_vasp/high_density_defects/{BP,GaSe,hBN,InSe}_spin*
git add datasets/raw_vasp/high_density_defects/{MoS,WSe}2_500
git add datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110/*.tar.gz
git commit -m "Add raw VASP files"
git push
```
Run the workflows in the following order. Don't forget to copy the files to git after each workflow.
1. Low density index
2. VASP to csv_cif
3. csv_cif to dataframe
4. Matminer

## Hyperparameter optimisation
### Get the data
```
dvc pull -R processed-high-density processed-low-density datasets/processed/{high,low}_density_defects datasets/experiments/combined_mixed_weighted_test.dvc datasets/experiments/combined_mixed_weighted_validation.dvc
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
### Rolos
Not implemented. If you really want to redo the step and have a couple of GPU-months to spare, create workflows like in the next step.
## Run the experiment on train/test split
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
### Rolos
Run the following workflows:
1. Combined test SchNet
2. Combined test CatBoost
3. Combined test MegNet full
4. Combined test MegNet sparse
They are independent, so you can copy the results to git once they all are done.
## Ablation study
Manually prepare the model configurations (aka trials) in `trials/megnet_pytorch/ablation_study`. Put them into a `.txt` and run the experiments:
```bash
cd scripts/ASPIRE-1
xargs ablation_stability.txt -L1 ./run_stability_trials.sh
```
Not implemented on Rolos. If you really want to redo the step and have GPU to spare, create workflows like in the previous step.
## MoS2 E(distance)
```bash
cd scripts/ASPIRE-1
xargs MoS2_V2_E.txt -L1 ./run_stability_trials.sh
```
Not implemented on Rolos. If you really want to redo the step and have GPU to spare, create workflows like in the previous step.
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