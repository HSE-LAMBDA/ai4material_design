import argparse
import pandas as pd
import numpy as np
import getpass
from collections import defaultdict
import yaml
from summary_table import read_targets
import matplotlib.pyplot as plt
import logging
import sys
sys.path.append('.')
from ai4mat.data.data import StorageResolver, get_prediction_path, get_targets_path
 
def get_results(experiment_family, trial, drop_zero):
    storage_resolver = StorageResolver()
    with open(storage_resolver["experiments"].joinpath(experiment_family, "family.txt")) as family_file:
        experiments = family_file.read().splitlines()

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # target, experiment_type, mae/in_domain_train_size, list of numbers
    for experiment_name in experiments:
        _, in_domain_train_size, experiment_type = experiment_name.split("/")
        if drop_zero and int(in_domain_train_size) == 0:
            continue

        experiment_path = storage_resolver["experiments"].joinpath(experiment_name)
        with open(experiment_path.joinpath("config.yaml")) as experiment_file:
            experiment = yaml.safe_load(experiment_file)
        
        folds = pd.read_csv(experiment_path.joinpath("folds.csv"),
                            index_col="_id",
                            squeeze=True)
        true_targets = pd.concat([pd.read_csv(get_targets_path(path),
                                  index_col="_id",
                                  usecols=["_id"] + experiment["targets"])
                                  for path in experiment["datasets"]], axis=0)
        
        for target in experiment["targets"]:
            try:
                predictions = pd.read_csv(storage_resolver["predictions"].joinpath(
                                        get_prediction_path(
                                        experiment_name,
                                        target,
                                        trial
                                        )), index_col="_id", squeeze=True)
                these_targets = true_targets.loc[:, target].reindex(index=predictions.index)
                errors = np.abs(predictions - these_targets)     
                results[target][experiment_type]["mae"].append(errors.mean())
                results[target][experiment_type]["in_domain_train_size"].append(int(in_domain_train_size))
            except FileNotFoundError:
                logging.warning(f"No predictions for {experiment_name}/{target}/{trial}")
    return results
    

def main():
    parser = argparse.ArgumentParser("Plots transfer learning experiments")
    parser.add_argument("--experiment-families", type=str, nargs="+")
    parser.add_argument("--plots-prefix", type=str, required=True)
    parser.add_argument("--trial", type=str, required=True)
    parser.add_argument("--unit-multiplier", type=float, default=1e3,
        help="As of May 2022, data in the project are in eV, so the 1000 mutliplier makes it meV")
    parser.add_argument("--drop-zero", action="store_true",
                        help="Drop experiments with zero in-domain training examples")
    parser.add_argument("--unit", default="meV", help="Unit of the target values after scaling.")
    parser.add_argument("--file-type", default="pdf")
    args = parser.parse_args()
    
    results = []
    for this_family in args.experiment_families:
        results.append(get_results(this_family, args.trial, args.drop_zero))
    
    combined_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # target, experiment_type, in_domain_train_size, list of maes
    for family_results in results:
        for target, target_results in family_results.items():
            for experiment_type, experiment_results in target_results.items():
                for mae, in_domain_train_size in zip(experiment_results["mae"], experiment_results["in_domain_train_size"]):
                    combined_results[target][experiment_type][in_domain_train_size].append(mae)

    storage_resolver = StorageResolver()
    error_kw = {'capsize': 5, 'capthick': 1}
    experiment_labels = {
        "in_domain": "Train on $\mathregular{WSe_2}$",
        "in_and_out_domain": "Train on $\mathregular{WSe_2}$ and $\mathregular{MoS_2}$",
    }
    target_labels = {
        "band_gap": "Band gap",
        "formation_energy_per_site": "Formation energy per site"
    }
    for target, target_results in combined_results.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for experiment_type, experiment_results in target_results.items():
            mean_maes = np.fromiter(map(np.mean, experiment_results.values()),
                                    count=len(experiment_results.values()),
                                    dtype=float)*args.unit_multiplier
            std_maes = np.fromiter(map(np.std, experiment_results.values()),
                                   count=len(experiment_results.values()),
                                   dtype=float)*args.unit_multiplier
            ax.errorbar(experiment_results.keys(), mean_maes, yerr=std_maes,
                        label=experiment_labels[experiment_type], marker="o", **error_kw)
        ax.set_xlabel("# of $\mathregular{WSe_2}$ in train")
        ax.set_ylabel(f"{target_labels[target]} MAE, {args.unit}")
        ax.legend()
        
        if args.file_type == "pdf":
            metadata = {
                "Title": f"Predictions for {target} "
                         f"for experiment families {', '.join(args.experiment_families)}",
                "Keywords": "2D materials, machine learning, graph neural network, transfer learning"}
            try:
                metadata["Author"] = getpass.getuser()
            except:
                pass
        else:
            metadata = None

        plots_folder = storage_resolver["plots"].joinpath(
                                args.plots_prefix,
                                target)
        plots_folder.mkdir(exist_ok=True, parents=True)
        fig.savefig(plots_folder.joinpath(f"{args.trial}.{args.file_type}"),
                    dpi=300,
                    bbox_inches="tight",
                    metadata=metadata)

if __name__ == "__main__":
    main()