# Reproducing "Sparse representation for machine learning the properties of defects in 2D materials"
## Setting up the environment
Packages are installed out-of-the-box. The terminal commands in the general documentation often assume the working folder to be `ai4material_design`, `cd` to it if needed. By default, WanDB integration is disabled, to optionally enable it, set you WanDB API key in [`scripts/Rolos/wandb_config.sh`](../scripts/Rolos/wandb_config.sh), commit and push. Note that if you add collaborators to your project, they will have access to your API key.
## Workflow important note
After running a workflow, you need to grab the outputs from the workflow and add them to git:
```bash
export WORKFLOW="<workflow name>"
# Example:
# export WORKFLOW="4 Combined test MegNet sparse"
cp -r "rolos_workflow_data/${WORKFLOW}/current/data/datasets" ai4material_design/
git add ai4material_design/datasets
git commit -m "Workflow ${WORKFLOW} results"
git push
```
## Note: the data are already here
The results of all the steps are already available in the repository, you can selectively reproduce the parts you want.
## Data preprocessing
### VASP -> csv/cif
To reduce the repository size, raw VASP files are not stored on the platform, you need to download them from DVC. Prior to that, you need to increase the project size, 100 Gb should be sufficient.
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
Run the workflows in the following order.
* `1 Low density index` creates technical files needed to preserve the historical structure indexing. Location: [`ai4material_design/datasets/csv_cif/low_density_defects_Innopolis-v1/{MoS2,WSe2}`](../datasets/csv_cif/low_density_defects_Innopolis-v1).
* `2 VASP to csv_cif` extracts the energy and band gap information from the raw VASP output. Location: [`ai4material_design/datasets/csv_cif/{high,low}_density_defects/*`](../datasets/csv_cif).
### csv/cif -> dataframe
Workflow `3 csv_cif to dataframe` converts the structures from standard [CIF](https://www.iucr.org/resources/cif) format to a fast platform-specific. It also preprocesses the target values, e. g. computes the formation energy per atom. Finally, it produces the sparse defect-only representations. Location: [`ai4material_design/datasets/processed/{high,low}_density_defects/*/{targets.csv,data.pickle}.gz`](../datasets/processed).
### csv/cif -> matminer
Workflow `3 Matminer` computes [matminer](https://github.com/hackingmaterials/matminer) descriptors, to be used with [CatBoost](https://catboost.ai/). Assuming the resources are available, the step takes around 3 days, you can skip it if don't plan on running CatBoost. Location: [`ai4material_design/datasets/processed/{high,low}_density_defects/*/matminer.csv.gz`](../datasets/processed).
## Computational experiments
We have prepared the the workflows that reproduce the tuned models evaluation. They train the models and produce predictions on the test dataset. Training is done 12 times with different random seeds and initializations to estimate the uncertainty. Run them concurrently:
* `4 Combined test SchNet`
* `4 Combined test CatBoost`
* `4 Combined test MegNet full`
* `4 Combined test MegNet sparse`

Location: [`ai4material_design/datasets/predictions/combined_mixed_weighted_test/**`](../datasets/predictions/combined_mixed_weighted_test).

For the rest of computations in the paper, you need to create the corresponding workflows. 
## Results analysis
The notebooks are used as a source for Rolos publications, to update go to the "Publications" tab, click "Synchronize" and "Publish"
* Aggregate performance tables [`ai4material_design/notebooks/Results tables.ipynb`](../notebooks/Results%20tables.ipynb)
* Quantum oscillation predictions [`ai4material_design/notebooks/MoS2_V2_plot.ipynb`](../notebooks/MoS2_V2_plot.ipynb)

## Regenerating platform-specific scripts
The scripts and workflows are already on the platform. This sec
### Data preprocessing
1. Generate the platform scripts from DVC
```bash
cd scripts/Rolos
./scripts/Rolos/generate_workflow_scrtipts_from_dvc.sh 8
```
2. Create the workflows
 - Create the workflows manually using the UI
 - Put your workflow and project ids to [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js)
 - Log in to the platform, open the browser console, paste the relevant parts from [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js). You need to do it for each workflow.
### Computational experiments
1. Generate the scripts:
```bash
cd scripts/Rolos
xargs -a stability_trials.txt -L1 ./generate_experiments_workflow.sh 
```
2. Create the workflows
 -  Create the workflows manually using the UI
 - Put your workflow and project ids to [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js)
 -  Log in to the platform, open the browser console, paste the relevant parts from [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js). You need to do it for each workflow.