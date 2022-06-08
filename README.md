# AI for material design
- The overall design is documented in the [flowchart](https://miro.com/welcomeonboard/eUdTWFNlaTZOZkc3NUlqd2o0TXB2QUUxRjFWVGxVcGtrWTJ5U01lbFZ1aFZxTFJRcUNyNG5NMjFaZkZ4S3pHRXwzMDc0NDU3MzU5MDMzOTQ0ODgx?invite_link_id=740759716756)
- Some design decisions are outlined in [RFC](https://docs.google.com/document/d/1Cc3772US-E73yQEMFn444OY9og9blKHpuP21sv9Gdxk/edit?usp=sharing)
- Project log is in [Notion](https://www.notion.so/AI-for-material-design-1f8f321d2ac54245a7af410d838929ae)
- Paper in [Overleaf](https://www.overleaf.com/project/61893015795e7b18e7979f53)

## Setting up the environment on slurm cluster

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



## Setting up the environment
[Install Poetry](https://python-poetry.org/docs/#installation)
```
poetry shell
poetry install
```

[Log in to WanDB](https://docs.wandb.ai/ref/cli/wandb-login)
## Running the pipepline
Below we descrbie a lightweight test run. The commands are assumed to be ran inside the poetry shell.

0. Pull the inputs from DVC
```
dvc pull datasets/csv_cif/pilot.dvc datasets/experiments/pilot-plain-cv.dvc
```

1. Prepare splits for experiments. Splits are shared between people, so don't overwrite them.
```
python scripts/prepare_data_split.py --datasets=datasets/csv_cif/pilot --experiment-name=pilot-plain-cv
```
This creates the experiment definition in `datasets/experiments/pilot-plain-cv`

2. Preprocess the data to get targets, pickled full and sparse structures
```
python scripts/parse_csv_cif.py --input-name=pilot
```
This creates `datasets/processed/pilot/{data.pickle.gzip,targets.csv}`

3. Run the experiments
Make sure you are logged in to WanDB and use WanDB entity you have access to. Adjust the `gpus` option to the GPUs you have
```
python run_experiments.py --experiments pilot-plain-cv --trials megnet-sparse-pilot --gpus 0 1 2 3 --wandb-entity hse_lambda
```
or if you want to submit it as slurm job then modify `slurm-job.sh` with the desired argument and export the required enviroment variables
then run
```
./slurm-job.sh
```

This creates predictions in `datasets/predictions/pilot-plain-cv` and run information at [WanDB](https://wandb.ai/hse_lambda/ai4material_design).

4. Plot the plots
```
python scripts/plot.py --experiments pilot-plain-cv --trials megnet-sparse-pilot
```
This produces plots in `datasets/plots/pilot-plain-cv`

### Running catboost

0. Pull the inputs from DVC
```
dvc pull datasets/csv_cif/pilot.dvc datasets/experiments/matminer-test.dvc
```

1. Prepare the targets and matminer features  
Can be done with one of the two following commands:  
Compute features on the machine (up to several minutes per structure on single core)
```
python scripts/compute_matminer_features.py --input-name=pilot --n-proc 8
```
OR load existing features
```
dvc pull datasets/processed/pilot.dvc
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
