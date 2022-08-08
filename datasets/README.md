# Datasets
## Structure
1. `raw_vasp` contains raw VASP outputs
2. `csv_cif` contains data in the Innopolis format
3. `processed` contains precomputed matminer features and targets along with piclked structures that
   are ignored by DVC. Each fodler here corresponds to a folder in `csv_cif`
4. `experiments` contains the definitions of the data splits for different experiments
5. `predictions` contains predictions from different models
6. `plots` contains plots with the results
7. `POSCARs` contains initial structures in POSCAR format, with metadata similar to `csv_cif`
8. `others` contains situational files that don't fall into above categories

## Datasets
### `dichalcogenides_x1s6_202109`
MoS2, WSe2; 4x4, 5x5 and 6x6 supercells; VASP; computed by Pengru and converted by Ruslan & Maxim
### `dichalcogenides_x1s6_202109_MoS2`
MoS2 sampled from `dichalcogenides_x1s6_202109`
### `dichalcogenides_innopolis_202105`
relaxed structures for MoS2, WS2, MoSe2, WSe2 defects with 4x4, 5x5 and 6x6 supercells, GPAW, 2021-05
### `dichalcogenides8x8_innopolis_202108`
relaxed structures for MoS2 defects in 8x8 supercell (~100 defects optimized out of 500 generated) GPAW, 2021-08

## DVC Cheatsheet:
This directory contains datasets versioned by DVC. 
Git stores metadata in `.dvc` files, the files are saved in Azure storage.

DVC reference: https://dvc.org/doc/command-reference/

To add a new directory:
- create `new_directory` in the corresponding location
- `dvc add new_directory` and `dvc commit`
- add metadata to git: `git add new_directory.dvc` and `git commit`
- push changes to git and Azure: `git push` and `dvc push`
- **NB** update README.md with brief explanation of `new_directory` contents

get data:
- `dvc pull <directory_name>`

## Example:

```
cd ai4material_design/datasets/raw
mkdir raw_pengru_202104_MoS2-vacancies
mv /home/jupyter/mnt/a.tgz raw_pengru_202104_MoS2-vacancies/vacancies_MoS2.tgz
dvc add raw_pengru_202104_MoS2-vacancies/
dvc commit
git add raw_pengru_202104_MoS2-vacancies.dvc datasets/.gitignore
git commit -m "DFT from Pengru" -a
git push
dvc push
```

