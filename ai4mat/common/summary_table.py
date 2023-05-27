from typing import Dict, Tuple, List
from functools import partial
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from prettytable import PrettyTable as pt
import re
from collections import defaultdict, OrderedDict
from ai4mat.data.data import read_trial

def print_table_paper(dataframe: pd.DataFrame,
                      trial_re,
                      model_names: OrderedDict,
                      format_: str = "latex"):
    """
    Vastly hardcoded table formatter for the paper
    Args:
    ....
    format: one of ("latex", "jupyter")
    """
    row_index = "dataset"
    column_index = "trial"
    separate_by = "target"
    all_separators = dataframe.index.get_level_values(separate_by).unique()
    individual_dataset_parser = re.compile(r"(?P<density>.+)_density_defects/(?P<material>[a-zA-Z0-9]+)")
    for table_index in all_separators:
        table_data = dataframe.xs(table_index, level=separate_by)
        records = []
        for row_name in sorted(table_data.index.get_level_values(row_index).unique()):
            if row_name == "combined":
                records_per_dataset = {"Material": "combined", "Density": "both"}
            else:
                match = individual_dataset_parser.match(row_name)
                material = match.group("material")
                if material == "hBN":
                    material = "h-BN"
                records_per_dataset = {
                    "Material": '\ce{'+ material + '}' if format_ == "latex" else material,
                    "Density": match.group("density")}
            for trial, data in table_data.xs(row_name, level=row_index).iterrows():
                if trial_re is not None:
                    model_name = trial_re.match(trial).group("name").replace("_pytorch", "")
                else:
                    model_name = trial
                if format_ == "pandas_separate_std":
                    records_per_dataset[model_name] = data['mae']
                    records_per_dataset[f"{model_name}_std"] = data['std']
                else:
                    records_per_dataset[model_name] = (data['mae'], data['std'])
                #norm_model = "stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45"
                #norm_model = "stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496"
                #norm = table_data.at[(row_name, norm_model), "mae"]
                #records_per_dataset[model_name] = data['mae']/norm
            records.append(records_per_dataset)
        table = pd.DataFrame.from_records(records)
        table.set_index(["Material", "Density"], inplace=True)
        if format_ == "pandas_separate_std":
            augmented_model_names = dict()
            for short_name, long_name in model_names.items():
                augmented_model_names[short_name] = f"{long_name} MAE"
                augmented_model_names[f"{short_name}_std"] = f"{long_name} std"
        else:
            augmented_model_names = model_names
        table = table[list(augmented_model_names.keys())]
        table.rename(columns=augmented_model_names, inplace=True)
        # ties are to handled manually
        if format_ == "latex":
            styled_table = table.style.highlight_min(axis=1, props='bfseries: ;')#.format(partial(format_result, latex_siunitx=True))
            print(styled_table.to_latex(column_format="cc|ccccc", hrules=True, siunitx=True))
        elif format_ in {"pandas", "pandas_separate_std"}:
            return table
        else:
            raise ValueError("Unknown format")


def print_tables(series, separate_by, column_format_re=None, row_format_re=None):
    table_keys = [name for name in series.index.names if name not in separate_by]
    rows, columns = table_keys
    all_separators = series.index.get_level_values(separate_by).unique()

    for table_index in all_separators:
        table_data = series.xs(table_index, level=separate_by)
        # Add None for missing values
        new_index = pd.MultiIndex.from_product(table_data.index.remove_unused_levels().levels)
        table_data = table_data.reindex(new_index)
        mae_table = pt()
        column_names = list(table_data.index.get_level_values(columns).unique())
        if column_format_re is not None:
            column_names = [column_format_re.match(name).group("name") for name in column_names]
        mae_table.field_names = [rows] + column_names
        for row_name in sorted(table_data.index.get_level_values(rows).unique()):
            if row_format_re is not None:
                table_row = [row_format_re.match(row_name).group("name")]
            else:
                table_row = [row_name]
            for column_name, cell_value in table_data.xs(row_name, level=rows).items():
                table_row.append(cell_value)
            mae_table.add_row(table_row)
        print(table_index)
        print(mae_table)


def format_result(row, latex_math=False, latex_siunitx=False):
    if isinstance(row, pd.Series):
        mae = row['mae']
        std = row['std']
    else:
        mae, std = row
    digits = int(max(-np.log10(std) + 1, 0))
    if latex_math:
        return f"${mae:.{digits}f} \\pm {std:.{digits}f}$"
    elif latex_siunitx:
        return f"{mae:.{digits}f}({std:.{digits}f})"
    else:
        return f"{mae:.{digits}f} ± {std:.{digits}f}"

def get_argparser():
    parser = argparse.ArgumentParser("Makes a text table with MAEs")
    parser.add_argument("--experiments", type=str, nargs="+", required=True)
    parser.add_argument("--trials", type=str, nargs="*")
    parser.add_argument("--stability-trials", type=str, nargs="*")
    parser.add_argument("--print-std", action="store_true")
    parser.add_argument("--targets", type=str, nargs="+")
    parser.add_argument("--column-format-re", type=re.compile,
                        help="Regular expression to be matched against the column names for formating.")
    parser.add_argument("--row-format-re", type=re.compile,
                        help="Regular expression to be matched against the row names for formating.")
    parser.add_argument("--separate-by", type=str,
        help="Tables are 2D, we must slice the data")
    parser.add_argument("--skip-missing-data", action="store_true",
                        help="Skip experiments that don't have data for all targets")
    parser.add_argument("--save-pandas", type=Path,
                        help="Save the pandas dataframe to a file")
    parser.add_argument("--bootstrap-significance", action="store_true",
                        help="Use bootstrap to estimate whether the differenecs are statistically significant")
    parser.add_argument("--multiple", type=float, default=1.0,
                        help="Multiply the results by a constant, e.g. to convert from eV to meV")
    paper_args = parser.add_mutually_exclusive_group()
    paper_args.add_argument("--paper-results", action="store_true")
    paper_args.add_argument("--paper-ablation-energy", action="store_true")
    paper_args.add_argument("--paper-ablation-homo-lumo", action="store_true")
    parser.add_argument("--prediction-storage-root", type=Path)
    parser.add_argument("--format", type=str, default="latex")
    return parser


def do_table(args):
    results = []
    for experiment in args.experiments:
        results = []
        if args.trials is not None:
            for trial in args.trials:
                results.append(read_trial(experiment, trial, args.skip_missing_data, args.targets,
                                          prediction_storage_root=args.prediction_storage_root))
        if args.stability_trials is not None:
            for trial_prefix in args.stability_trials:
                results_for_stabiliy_family = []
                for trial_index in range(1, 13):
                    trial = f"{trial_prefix}/{trial_index}"
                    results_for_stabiliy_family.append(
                        read_trial(experiment, trial, args.skip_missing_data, args.targets,
                                   prediction_storage_root=args.prediction_storage_root))
                these_results_pd = pd.concat(results_for_stabiliy_family, axis=0)
                combined_results = []
                for (target, dataset), stability_results in these_results_pd.groupby(level=["target", "dataset"]):
                    combined_results.append({
                        "trial": trial_prefix,
                        "target": target,
                        "dataset": dataset,
                        "mae": stability_results["mae"].mean(),
                        "std": stability_results["mae"].std()
                    })
                combined_results_pd = pd.DataFrame.from_records(combined_results)
                combined_results_pd.set_index(["target", "dataset", "trial"], inplace=True)
                results.append(combined_results_pd)

    
    results_pd = args.multiple * pd.concat(results, axis=0)
    if args.save_pandas:
        results_pd.to_pickle(args.save_pandas)
        

    if args.paper_results:
        models=OrderedDict(
                schnet="SchNet",
                gemnet="GemNet",
                megnet="MEGNet",
                catboost="CatBoost")
        models["megnet/sparse"] = "Sparse (MEGNet)"
        return print_table_paper(results_pd, args.column_format_re, models, format_=args.format)
    elif args.paper_ablation_energy:
        models=OrderedDict()
        models["stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7"] = "Full"
        models["stability/megnet_pytorch/ablation_study/d6b7ce45-sparse"] = "Sparse"
        models["stability/megnet_pytorch/ablation_study/d6b7ce45-sparse-z"] = "Sparse-Z"
        models["stability/megnet_pytorch/ablation_study/d6b7ce45-sparse-z-were"] = "Sparse-Z-Were"
        models["stability/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45"] = "Sparse-Z-Were-EOS"
        return print_table_paper(results_pd, args.column_format_re, models, format_=args.format)
    elif args.paper_ablation_homo_lumo:
        models=OrderedDict()
        models["stability/megnet_pytorch/25-11-2022_11-38-18/1baefba7"] = "Full"
        models["stability/megnet_pytorch/ablation_study/831cc496-sparse"] = "Sparse"
        models["stability/megnet_pytorch/ablation_study/831cc496-sparse-z"] = "Sparse-Z"
        models["stability/megnet_pytorch/ablation_study/831cc496-sparse-z-were"] = "Sparse-Z-Were"
        models["stability/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496"] = "Sparse-Z-Were-EOS"
        return print_table_paper(results_pd, args.column_format_re, models, format_=args.format)
    else:
        results_str = results_pd.apply(lambda row: f"{row['mae']:.{4-int(np.log10(args.multiple))}f}", axis=1)
        print_tables(results_str, args.separate_by, args.column_format_re, args.row_format_re)

    if args.bootstrap_significance:
        if args.stability_trials:
            raise NotImplementedError("Bootstrap significance not implemented for stability trials")
        # Select the subsets that for each target and dataset
        bootstrap_results = []
        for (target, dataset), these_results in results_pd.groupby(level=["target", "dataset"]):
            ranks = []
            this_errors = np.stack(these_results.errors.values)
            this_weights = np.stack(these_results.weights.values)
            for i in range(1, this_weights.shape[0]):
                assert np.allclose(this_weights[0], this_weights[i])
            for _ in range(1000):
                boostrap_selection = np.random.choice(
                    this_errors.shape[1], this_errors.shape[1], replace=True, p=this_weights[0]/this_weights[0].sum())
                maes = np.average(this_errors[:, boostrap_selection], axis=1)
                ranks.append(np.argsort(maes))
            ranks = np.stack(ranks)
            for (_, _, trial), rank in zip(these_results.index, ranks.T):
                these_ranks, these_counts = np.unique(rank, return_counts=True)
                bootstrap_results.append({
                    "trial": trial,
                    "target": target,
                    "dataset": dataset,
                    "ranks": these_ranks,
                    "counts": these_counts,
                })
        bootstrap_results_pd = pd.DataFrame.from_records(bootstrap_results)
        bootstrap_results_pd.set_index(["target", "dataset", "trial"], inplace=True)
        bootstrap_results_pd_str = bootstrap_results_pd.apply(lambda row: str(row["ranks"]) + str(row["counts"]), axis=1)
        print_tables(bootstrap_results_pd_str, args.separate_by, args.column_format_re, args.row_format_re)