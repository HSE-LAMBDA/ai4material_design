# data folder

This directory contains datasets versioned by DVC. 
Git stores metadata in `.dvc` files, the files are saved in Azure storage. 

## Contents
- `raw_ruslan_202104` - relaxed states, relaxation trajectories, electronic properties, GPAW, 2021-04
- `processed_ruslan_202104` - processed table of the above (without the relaxation trjectory), GPAW, 2021-04
- `raw_pengru_202104_MoS2-vacancies` - relaxation (?) of MoS2 lattices, VASP, 2021-04
- `dichalcogenides_innopolis_202105` - relaxed structures for MoS2, WS2, MoSe2, WSe2 defects with 4x4, 5x5 and 6x6 supercells, GPAW, 2021-05

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
