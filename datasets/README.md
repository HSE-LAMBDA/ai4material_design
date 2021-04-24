# data folder

This directory contains datasets versioned by DVC. 
Git stores metadata in `.dvc` files, the files are saved in Azure storage. 

## Cheatsheet:

add new directory:
- create `new_directory` with datasets
- `dvc add new_directory` and `dvc commit`
- add metadata to git: `git add new_directory.dvc` and `git commit`
- push changes to git and Azure: `git push` and `dvc push`

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
