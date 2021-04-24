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
