# Data preprocessing: VASP -> csv/cif -> pickle & matminer
Run the workflows in the following order. Same number means the workflows can be run concurrently.
* `1 Low density index` creates technical files needed to preserve the historical structure indexing. Location: [`ai4material_design/datasets/csv_cif/low_density_defects_Innopolis-v1/{MoS2,WSe2}`](../datasets/csv_cif/low_density_defects_Innopolis-v1).
* `2 VASP to csv_cif` extracts the energy and band gap information from the raw VASP output. Location: [`ai4material_design/datasets/csv_cif/{high,low}_density_defects/*`](../datasets/csv_cif).
* `3 csv_cif to dataframe` converts the structures from standard [CIF](https://www.iucr.org/resources/cif) format to a fast platform-specific. It also preprocesses the target values, e. g. computes the formation energy per atom. Finally, it produces the sparse defect-only representations. Location: [`ai4material_design/datasets/processed/{high,low}_density_defects/*/{targets.csv,data.pickle}.gz`](../datasets/processed).
* `3 Matminer` computes [matminer](https://github.com/hackingmaterials/matminer) descriptors, to be used with [CatBoost](https://catboost.ai/). Location: [`ai4material_design/datasets/processed/{high,low}_density_defects/*/matminer.csv.gz`](../datasets/processed).
# Computational experiments
Run the following workflows, they can be ran concurrently:
* `4 Combined test SchNet`
* `4 Combined test CatBoost`
* `4 Combined test MegNet full`
* `4 Combined test MegNet sparse`

They train the models and produce predictions on the test dataset. Training is done 12 times with different random seeds and initializations to estimate the uncertainty. Location: [`ai4material_design/datasets/predictions/combined_mixed_weighted_test/**`](../datasets/predictions/combined_mixed_weighted_test).
# Results analysis
The notebooks are used as a source for Rolos publications, to update go to the "Publications" tab, click "Synchronize" and "Publish"
* Aggregate performance tables [`ai4material_design/notebooks/Results tables.ipynb`](../notebooks/Results%20tables.ipynb)
* Quantum oscillation predictions [`ai4material_design/notebooks/MoS2_V2_plot.ipynb`](../notebooks/MoS2_V2_plot.ipynb)
# Notes
The workflows in this project are mock workflows for demonstration purposes. The code and data are real, of course. The version of the project with all the workflows enabled is [here](https://my.rolos.com/public/project/79a29e5d84da4e5680ed6d8c9f933748). It requires around 16 GPU days to run, so it is not feasible to execute it on a free trial account.