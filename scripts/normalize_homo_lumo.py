import pandas as pd
import argparse

from pathlib import Path
import sys

sys.path.append('.')
from ai4mat.data.data import StorageResolver



def main():
    parser = argparse.ArgumentParser("Normalize HOMO and LUMO")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions-folder", type=str)
    group.add_argument("--dataset-folder", type=str, default='dichalcogenides_x1s6_202109_MoS2')
    args = parser.parse_args()

    storage_resolver = StorageResolver()

    structures = pd.read_pickle(storage_resolver['processed'] /  args.dataset_folder / 'data.pickle.gz')
    structures['norm_const'] = structures['homo'] - structures['normalized_homo'] 


    predictions = (storage_resolver['predictions'] / args.predictions_folder).rglob('*.csv.gz')
    predictions = list(
        filter(
            lambda item: 'homo' in item.parent.name or 'lumo' in item.parent.name,
            predictions
            )
        )
    for file in predictions:
        pred = pd.read_csv(file, index_col=0)
        if len(pred.columns) >= 2: continue
        pred = pred.iloc[:, 0] - structures['norm_const']
        save_dir = file.parents[1] / f'normalized_{file.parents[0].name}'
        if not save_dir.exists():
            save_dir.mkdir()

        pred.to_csv(
            save_dir / file.name
        )


if __name__ == '__main__':
    main()