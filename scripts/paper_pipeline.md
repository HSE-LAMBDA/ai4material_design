1. Prepare splits for experiments
``
poetry run python scripts/prepare_data_split.py --datasets=datasets/csv_cif/dichalcogenides_x1s6_202109 --output-folder=datasets/experiments/paper-2-plain-cv
```
2. Preprocess the data to get targets, pickled full and sparse structures
```
poetry run python scripts/parse_csv_cif.py --input-folder=datasets/csv_cif/dichalcogenides_x1s6_202109 --output-folder=datasets/processed/dichalcogenides_x1s6_202109 --ignore-missing
```
3. Run the experiments
```
poetry run python scripts/run_experiments.py --experiments datasets/experiments/paper-2-plain-cv --trials trials/megnet-sparse-pilot.yml --gpus=0 1 2 3  --predictions-root datasets/predictions
```
