from itertools import product
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import getpass
import matplotlib.pyplot as plt

from data import (
    StorageResolver,
    get_prediction_path,
    get_targets_path,
    NICE_TARGET_NAMES
)


def main():
    parser = argparse.ArgumentParser("Plots predictions for two trials")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--trials", type=str, nargs="+")
    parser.add_argument("--file-type", type=str, default="pdf")
    args = parser.parse_args()
    
    storage_resolver = StorageResolver()
    for experiment_name in args.experiments:
        experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
        with open(experiment_path.joinpath("config.yaml")) as experiment_file:
            experiment = yaml.safe_load(experiment_file)
        folds = pd.read_csv(experiment_path.joinpath("folds.csv"),
                            index_col="_id",
                            squeeze=True)
        # Support running on a part of the dataset, defined via folds
        true_targets = pd.concat([pd.read_csv(get_targets_path(path), index_col="_id")
                                  for path in experiment["datasets"]], axis=0).reindex(
                                          index=folds.index)
        for target_name in experiment["targets"]:
            fig, axes = plt.subplots(ncols=len(args.trials))
            first_plot = True
            for this_trial_name, ax in zip(args.trials, axes):
                predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
                    get_prediction_path(
                        experiment_name,
                        target_name,
                        this_trial_name
                    )), index_col="_id", squeeze=True)
                assert predictions.index.equals(true_targets.index)
                mae = np.abs(predictions - true_targets.loc[:, target_name]).mean()
            
                ax.scatter(true_targets.loc[:, target_name],
                           predictions,
                           label=f"$\mathrm{{MAE}}={mae:.4f}$",
                           alpha=0.5)
                ax.set_xlabel(f"DFT {NICE_TARGET_NAMES[target_name]}")
                if first_plot:
                    ax.set_ylabel(f"Predicted {NICE_TARGET_NAMES[target_name]}")
                    first_plot = False
                ax.legend()
                with open(storage_resolver["trials"].joinpath(
                        f"{this_trial_name}.yaml")) as this_trial_file:
                    this_trial = yaml.safe_load(this_trial_file)
                ax.set_title(f"{this_trial['model']}, {this_trial['representation']} representation")

                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()]),
                ]
                ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
                ax.set_aspect('equal')
                ax.set_xlim(lims)
                ax.set_ylim(lims)

            plots_folder = storage_resolver["plots"].joinpath(
                experiment_name,
                target_name)
            plots_folder.mkdir(exist_ok=True, parents=True)
            metadata = {
                "Title": f"Predictions for {target_name} "
                         f"for experiment {experiment_name}",
                "Keywords": "2D materials, machine learning, graph neural network, MEGNet"}
            try:
                metadata["Author"] = getpass.getuser()
            except:
                pass
            fig.tight_layout()
            fig.savefig(Path(plots_folder,
                             f"combined_{'_'.join(args.trials)}.{args.file_type}"),
                        metadata=metadata, bbox_inches="tight")

if __name__ == "__main__":
    main()
