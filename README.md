# AI for material design
- The overall design is documented in the [flowchart](https://miro.com/welcomeonboard/eUdTWFNlaTZOZkc3NUlqd2o0TXB2QUUxRjFWVGxVcGtrWTJ5U01lbFZ1aFZxTFJRcUNyNG5NMjFaZkZ4S3pHRXwzMDc0NDU3MzU5MDMzOTQ0ODgx?invite_link_id=740759716756)
- Some design desision are outlined in [RFC](https://docs.google.com/document/d/1Cc3772US-E73yQEMFn444OY9og9blKHpuP21sv9Gdxk/edit?usp=sharing)
- Project log is in [Notion](https://www.notion.so/AI-for-material-design-1f8f321d2ac54245a7af410d838929ae)

## Running the pipepline
Below we descrbie a lightweight test run

0. Pull the inputs from DVC
```
dvc pull datasets/csv_cif/pilot.dvc datasets/experiments/pilot-plain-cv.dvc
```

1. Prepare splits for experiments. Splits are shared between people, so don't overwrite them.
```
poetry run python scripts/prepare_data_split.py --datasets=datasets/csv_cif/pilot --experiment-name=pilot-plain-cv
```
This creates the experiment definition in `datasets/experiments/pilot-plain-cv`

2. Preprocess the data to get targets, pickled full and sparse structures
```
poetry run python scripts/parse_csv_cif.py --input-name=pilot
```
This creates `datasets/processed/pilot/{data.pickle.gzip,targets.csv}`

3. Run the experiments
Make sure you are logged in to WanDB and use WanDB entity you have access to. Adjust the `gpus` option to the GPUs you have
```
poetry run python scripts/run_experiments.py --experiments pilot-plain-cv --trials megnet-sparse-pilot --gpus 0 1 2 3 --wandb-entity hse_lambda
```
This creates predictions in `datasets/predictions/pilot-plain-cv` and run information at [WanDB](https://wandb.ai/hse_lambda/ai4material_design).

4. Plot the plots
```
poetry run python scripts/plot.py --experiments pilot-plain-cv --trials megnet-sparse-pilot
```
This produces plots in `datasets/plots/pilot-plain-cv`

# Obsolete sections to be updated
## Predicting energy with CatBoost and matminer experiment
`catboost_experiment_reproduction/` contains scripts for the experiment reproduction.
`datasets/paper_experiments_catboost/` contains the data, generated during the experiment.   
I isolated them to simplify the structure of experiment, we can change it later.
* Prepare the matminer features for each defect. There are two ways for that:
  * Copy generated features from `datasets/dichalcogenides_innopolis_features/` to `datasets/paper_experiments_catboost/features/`
  * Run `python catboost_experiment_reproduction/make_features.py` (switch parameter `compute_all` to `True` before, otherwise you will run the default version that computes just 5 defects). 
* Copy the folds definitions from `datasets/paper_experiments/inputs/` to `datasets/paper_experiments_catboost/folds/`.
* Run the training with `python catboost_experiment_reproduction/train_model.py`. This should produce models in `datasets/paper_experiments_catboost/models/`.
* Get the predictions and the plot from `catboost_experiment_reproduction/catboost_predictions.ipynb`
* You now have `datasets/paper_experiments_catboost/results/full.csv.gz`

## Running on HSE HPC
* Clone the repo in the cluster
* Pull the data `dvc pull`
* Load singularity `module load singularity`
* Set `WANDB_ENTITY`, `WANDB_API_KEY` in `singularity.sbatch
* Get the image `singularity pull --docker-login  docker://abdalazizrashis/ai4material_design:latest`
* Preprocess the data by running `singularity run singularity run ai4material_design_latest.sif python Defect_representation.py`
* Submit the the jon `sbatch singularity.sbatch`
* You can run the job interactively `srun -G 1 -c 4 --pty bash` to get interactive shell the run whatever script i.e. `singularity run --nv ai4material_design_latest.sif python megnet_grahps_train.py`    
