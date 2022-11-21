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
4. [Install pytorch](https://pytorch.org/) according to your CUDA/virtualenv/conda situatoin
5. [Install pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) according to your CUDA/virtualenv/conda situatoin
6. [Log in to WanDB](https://docs.wandb.ai/ref/cli/wandb-login)

## Running the pilot pipepline
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

## Data transformation: DVC pipline
### Getting the data
The `.dvc` files are no longer there - but the data are!
```
dvc pull datasets/csv_cif/high_density_defects/{BP_spin,GaSe_spin,hBN_spin,InSe_spin,MoS2,WSe2}_500
dvc pull datasets/processed/high_density_defects/{BP_spin,GaSe_spin,hBN_spin,InSe_spin,MoS2,WSe2}_500/{targets.csv,data.pickle}.gz
dvc pull datasets/csv_cif/low_density_defects/{MoS2,WSe2}
dvc pull datasets/processed/low_density_defects/{MoS2,WSe2}/{targets.csv,data.pickle}.gz
```
### Reproducing the pipeline
VASP -> csv_cif -> processed -> Rolos has been implemented a [DVC pipeline](https://dvc.org/doc/start/data-management/data-pipelines). Processed datasets:
```
parallel --delay 3 -j6 dvc repro processed-high-density@{} ::: hBN_spin GaSe_spin BP_spin InSe_spin MoS2 WSe2
parallel --delay 3 -j2 dvc repro processed-low-density@{} ::: MoS2 WSe2
```
Archives for Rolos with structutres and targets:
```
dvc repro rolos-2d-materials-point-defects
```
Note that unlike GNU Make DVC [currently](https://github.com/iterative/dvc/issues/755) doesn't internally parallelize execution, so we use GNU parallel. We also use `--delay 3` to avoid [DVC lock race](https://github.com/iterative/dvc/issues/755).
## Running experiments with GNU parallel
Single experiment family:
```bash
parallel -a datasets/experiments/MoS2_to_WSe2/family.txt -j4 python run_experiments.py --experiments {1} --trials megnet_pytorch-sparse-10 --gpus 0 --wandb-entity hse_lambda
```
Multiple families:
```bash
awk 1 datasets/experiments/MoS2_to_WSe2_4?/family.txt | WANDB_RUN_GROUP="MoS2_to_WSe2 $(date --rfc-3339=seconds)" parallel -j4 python run_experiments.py --experiments {} --trials megnet_pytorch-sparse-10 --gpus 0 --wandb-entity hse_lambda :::: -
```
## Running CatBoost on pilot
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
dvc pull datasets/processed/pilot
```

Both scenarios produce `datasets/processed/pilot/matminer.csv.gz`

2. Run the experiments
```
python run_experiments.py --experiments matminer-test --trials catboost-test --gpus 0 1 2 3 --wandb-entity hse_lambda   
```
This creates predictions in `datasets/predcitions/matminer-test`

3. Plot the plots
```
python scripts/plot.py --experiments matminer-test --trials catboost-test
```
This produces plots in `datasets/plots/matminer-test`

## CatBoost for paper
1. Data preprocessing
Pull the csv/cif from dvc:
```
dvc pull datasets/csv_cif/high_density_defects/{BP_spin,GaSe_spin,hBN_spin,InSe_spin,MoS2,WSe2}_500 datasets/csv_cif/low_density_defects/{MoS2,WSe2}
```
then run the featuriser
```
parallel -j8 --delay 5 dvc repro matminer@{} ::: high_density_defects/{BP_spin,GaSe_spin,hBN_spin,InSe_spin,MoS2,WSe2}_500 low_density_defects/{MoS2,WSe2}
```
or, on ASPIRE 1
```
export SINGULARITYENV_PATH=/opt/conda/lib/python3.8/site-packages/torch_tensorrt/bin:/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin:/home/users/nus/kna/.local/bin 
export IMAGE=/home/projects/ai/singularity/nvcr.io/nvidia/pytorch:22.04-py3.sif
parallel -j8 --delay 10 singularity exec $IMAGE dvc repro matminer@{} ::: high_density_defects/{BP_spin,GaSe_spin,hBN_spin,InSe_spin,MoS2,WSe2}_500 low_density_defects/{MoS2,WSe2}
```
If DVC wants to recompute csv/cif, but you are sure it's not needed, you can add `--downstream` to the dvc arguments.
## Additional considerations
### `prepare_data_split.py`
Generates data splits aka experiments. There is no need to do this step to run the existing experiments, the splits are available in DVC. Splits are by design shared between people, so don't overwrite them needlessly. Example:
```
python scripts/prepare_data_split.py --datasets=datasets/csv_cif/pilot --experiment-name=pilot-plain-cv --targets band_gap homo formation_energy_per_site
```
This creates the experiment definition in `datasets/experiments/pilot-plain-cv`

It supports generating cross-validation and train/test splits.

### `parse_csv_cif.py`
For data computed without spin interaction, you might want to add `--fill-missing-band-properties` flag that would fill `{band_gap,homo,lumo}_{majority,minority}` from `{band_gap,homo,lumo}`. For old data you might also want to use the flag to fill `band_gap = lumo - homo` Example:
```
python scripts/parse_csv_cif.py --input-name high_density_defects/GaSe --fill-missing-band-properties
```

## Running random search for the paper
Get the data
```
dvc pull datasets/csv_cif/high_density_defects/{BP_spin,GaSe_spin,hBN_spin,InSe_spin,MoS2,WSe2}_500 datasets/csv_cif/low_density_defects/{MoS2,WSe2} datasets/processed/high_density_defects/{BP_spin,GaSe_spin,hBN_spin,InSe_spin,MoS2,WSe2}_500/{data.pickle.gz,targets.csv.gz} datasets/processed/low_density_defects/{MoS2,WSe2}/{data.pickle.gz,targets.csv.gz} datasets/experiments/combined_mixed_weighted_test.dvc datasets/experiments/combined_mixed_weighted_validation.dvc
```
### megnet_pytorch_sparse
```
dvc pull trials/megnet_pytorch/09-11-2022_18-11-54.dvc
./run_megnet_pytorch_experiments_nscc_paper.sh
```
### Print the tables [obsolete]
```
python scripts/summary_table_lean.py --experiments high_density/{BP,hBN,InSe,GaSe}_spin_500 high_density/{MoS2,WSe2}_500 high_density/combined --combined-experiment high_density/combined --trials megnet_pytorch_paper/sparse{,-z,-z-were,-z-were-eos} --skip-missing --separate-by target --column-format-re megnet_pytorch_paper/\(?P\<name\>.+\) --row-format-re high_density/\(?P\<name\>.\*\)
python scripts/summary_table_lean.py --experiments low_density/{MoS2,WSe2,combined} --combined-experiment low_density/combined --trials megnet_pytorch_paper/sparse{,-z,-z-were,-z-were-eos} --skip-missing --separate-by target --column-format-re megnet_pytorch_paper/\(?P\<name\>.+\) --row-format-re low_density/\(?P\<name\>.\*\)
python scripts/summary_table_lean.py --experiments low_density/{MoS2,WSe2,combined} high_density/{BP,hBN,InSe,GaSe}_spin_500 high_density/{MoS2,WSe2}_500 high_density/combined low_high_combined --trials megnet_pytorch_paper/sparse{,-z,-z-were,-z-were-eos} --skip-missing --separate-by target --save-pandas datasets/others/megnet_sparse_analysis.pkl.gz
```

## Running on HSE HPC [obsolete]
0. ssh to the cluster head node if you gonna run on a slurm cluster
1. Load the module `module load Python/Anaconda_v11.2020`
2. Install poetry ```curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -```
3. create new conda enviroment ```conda create -n exp1 python=3.8``` then activate it `conda activate exp1`
4. cd to project directory and run `poetry install` if you having internal poetry problem due to the fact you are already using poetry and didn't install it run ```pip install poetry```
5. find out cuda version, then `export CUDA=cu113` replace the `cu113` with your version
6. run
```
pip install torch==1.10.0+${CUDA} torchvision==0.11.1+${CUDA} torchaudio==0.10.0+${CUDA} -f https://download.pytorch.org/whl/${CUDA}/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-geometric
```
if you have error make sure to run and repeat the previous step
```
pip uninstall torch-geometric torch-sparse torch-scatter torch-spline-conv torch-cluster
```
this step is very ugly but this the fastest way to have a working enviroment
