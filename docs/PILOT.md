# Running the pilot NN model
Below we descrbie a lightweight test run.

0. Pull the inputs from DVC. On Rolos, they are already there.
```
dvc pull datasets/csv_cif/pilot datasets/experiments/pilot-plain-cv datasets/processed/pilot/{targets.csv,data.pickle}.gz
```

1. Preprocess the data to get targets, pickled full and sparse structures
```
python scripts/parse_csv_cif.py --input-name=pilot --fill-missing-band-properties
```
This creates `datasets/processed/pilot/{data.pickle.gzip,targets.csv}`

2. Run the experiments
This step creates predictions in `datasets/predictions/pilot-plain-cv` and run information at [WanDB](https://wandb.ai/hse_lambda/ai4material_design). Make sure you are logged in to WanDB and use WanDB entity you have access to, ot set `WANDB_MODE=disabled`.
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
# Running a pilot CatBoost model
0. Pull the inputs from DVC. As usual, on Rolos they are alaready available.
```
dvc pull datasets/csv_cif/pilot datasets/experiments/pilot-plain-cv
```

1. Prepare the targets and matminer features
```bash
python scripts/parse_csv_cif.py --input-name=pilot --fill-missing-band-properties
```
Computing matminer features takes several minutes per structure on single CPU core, and requires downgrading `numpy`, so you might want to just load the precomputed features (already done on Rolos):
```bash
dvc pull datasets/processed/pilot/matminer.csv.gz
```
If you want to compute them yourself, run
```bash
pip install "numpy<1.24.0"
python scripts/compute_matminer_features.py --input-name=pilot --n-proc 8
```
Both scenarios produce `datasets/processed/pilot/matminer.csv.gz`

2. Run the experiments
On 4 GPUs:
```bash
python run_experiments.py --experiments pilot-plain-cv --trials catboost/pilot --gpus 0 1 2 3 --wandb-entity hse_lambda
```
On 2 CPUs:
```bash
python run_experiments.py --experiments pilot-plain-cv --trials catboost/pilot --cpu --processes-per-unit 2 --wandb-entity hse_lambda
```
This creates predictions in `datasets/predictions/pilot-plain-cv/*/catboost`

3. Plot the plots
```bash
python scripts/plot.py --experiments pilot-plain-cv --trials catboost/pilot
```
This produces plots in `datasets/plots/matminer-test`