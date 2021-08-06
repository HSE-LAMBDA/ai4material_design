# data folder

This directory contains datasets versioned by DVC. 
Git stores metadata in `.dvc` files, the files are saved in Azure storage. 

## Contents - latest
### `dichalcogenides_innopolis_202105`
relaxed structures for MoS2, WS2, MoSe2, WSe2 defects with 4x4, 5x5 and 6x6 supercells, GPAW, 2021-05
### `predicted_dichalcogenides_innopolis_202105_v3.dvc`
MEGNet on defects predictions. Code corresponds to [this commit](29419a2f9e060226f732bfaba1c9b99296344209). The data were taken from `dichalcogenides_innopolis_202105` and filtered for formation energy > -4 eV/defect site. Summary of the experiments configurations and results is recorded in `summary.csv.gz`. For each experiment configuration,`{name}_train.csv` contains the training data, `{name}_test.csv` the testing. Columns: `_id`,`homo`,`energy_per_atom` correspond to `dichalcogenides_innopolis_202105/defects.csv`; `formation_energy` is computed as `defected structure energy - initial structure energy + free atoms energies`, with initial energy from `dichalcogenides_innopolis_202105/initial_structures.csv`; `predicted_homo`,`predicted_formation_energy` are the values predicted by MEGNet of defect graphs. The correspondence between the experiment name and conditions is in `summary.csv.gz`, for details see the [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/megnet_graphs_train.py) and the slides. The `plots` folder contains the plots for each experiment. Learning curves are [here](https://wandb.ai/kazeev/ai4material_design/groups/Defect-only-MEGNet-1m0zitqk/workspace?workspace=user-).

Best performing experiments suggested for expert analysis:
1. `formation_energy_full_bond_z_embed`
2. `formation_energy_vac_only_bond_z_Z`
3. `homo_full_bond_z_werespecies`
4. `homo_vac_only_bond_z_Z`

## Contents - historical
- `raw_ruslan_202104` - relaxed states, relaxation trajectories, electronic properties, GPAW, 2021-04
- `processed_ruslan_202104` - processed table of the above (without the relaxation trjectory), GPAW, 2021-04
- `predicted_dichalcogenides_innopolis_202105_v2.dvc` - MEGNet on defects predictions. Code corresponds to [this commit](https://github.com/HSE-LAMBDA/ai4material_design/tree/2de4d6751c10332fa8138734eb6941580670d11b). The data were taken from `dichalcogenides_innopolis_202105` and filtered for formation energy > 0 as a precatuion to avoid the yet unsolved issue with some structures. `*train.csv` contain the training data, `*test.csv` the testing. `clean_full` is based on the full dataset (3192 structures) and `clean_vac_only` is based on the dataset with only vacancies (3190 structures). Columns: `_id`,`homo`,`energy_per_atom` correspond to `dichalcogenides_innopolis_202105/defects.csv`; `formation_energy` is computed as `defected structure energy - initial structure energy + free atoms energies`, with initial energy from `dichalcogenides_innopolis_202105/initial_structures.csv`; `predicted_homo`,`predicted_formation_energy` are the values predicted by MEGNet of defect graphs.
- `predicted_dichalcogenides_innopolis_202105_v1.dvc` - OBSOLETE MEGNet on defects predictions. Code corresponds to [this commit](https://github.com/HSE-LAMBDA/ai4material_design/commit/a4018a49fbc5ac85f0c493eac90920cc17bbe01d). `train.csv` contains the training data, `test.csv` the testing. Columns: `_id`,`homo`,`energy_per_atom` correspond to `dichalcogenides_innopolis_202105/defects.csv`; `formation_energy` is computed as `defected structure energy - initial structure energy`, with initial energy from `dichalcogenides_innopolis_202105/initial_structures.csv`; `predicted_homo`,`predicted_formation_energy`,`predicted_energy_per_atom` are the values predicted by MEGNet of defect graphs. WARNING the structures with very low formation energy (`< -20 eV`) are most likely wrong, and all formation energies don't take into account the single atom energies.
- `raw_pengru_202104_MoS2-vacancies` - relaxation (?) of MoS2 lattices, VASP, 2021-04
## Cheatsheet:

add new directory:
- create `new_directory` with datasets
- `dvc add new_directory` and `dvc commit`
- add metadata to git: `git add new_directory.dvc` and `git commit`
- push changes to git and Azure: `git push` and `dvc push`
- **NB** update README.md with brief explanation of `new_directory` contents

get data:
- `dvc pull <directory_name>`

## Example:

```
cd ai4material_design/data/
mkdir raw_pengru_202104_MoS2-vacancies
mv /home/jupyter/mnt/a.tgz raw_pengru_202104_MoS2-vacancies/vacancies_MoS2.tgz
dvc add raw_pengru_202104_MoS2-vacancies/
dvc commit
git add raw_pengru_202104_MoS2-vacancies.dvc datasets/.gitignore
git commit -m "DFT from Pengru" -a
git push
dvc push
```

## References:
- DVC reference: https://dvc.org/doc/command-reference/
