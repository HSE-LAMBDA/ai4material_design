import argparse
import yaml
import hashlib
import json
import os

from itertools import product
from copy import deepcopy

from pathlib import Path
from plumbum import local, FG
from datetime import datetime
from tqdm import tqdm

from ai4mat.data.data import StorageResolver


def dfs(data, path, sets):
    """
    walk through nested dict and store all paths in format 'a b c'
    """
    for key, value in data.items():
        path.append(key)
        if type(value) == list:
            cur_set = []
            cur_path = " ".join(path)
            for val in value:
                cur_set.append((cur_path, val))
            sets.append(cur_set)
        else:
            dfs(value, path, sets)
        path = path[:-1]


def set_item_by_path(d, value, path):
    if len(path) == 1:
        d[path[-1]] = value
    else:
        set_item_by_path(d[path[0]], value, path[1:])


def generate_trials(template, param_config):
    param_sets = []
    dfs(param_config, [], param_sets)

    for params_change in product(*param_sets):
        cur_template = deepcopy(template)
        for param in params_change:
            set_item_by_path(cur_template, param[1], list(param[0].split()))
        yield cur_template


def parse_args():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--warm-start", type=str)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--mode", choices=['grid, random'], required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    storage_resolver = StorageResolver()

    if args.warm_start is None:
        template_path = storage_resolver['templates'].joinpath(args.model_name).joinpath("parameters_template.yaml")
        param_config_path = storage_resolver['templates'].joinpath(args.model_name).joinpath("parameters_to_tune.yaml")

        with open(template_path) as f:
            with open(param_config_path) as ff:
                template = yaml.safe_load(f)
                param_config = yaml.safe_load(ff)

        relative_dir_path = Path(args.model_name).joinpath(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        res_dir_path = storage_resolver['trials'].joinpath(relative_dir_path)
        mkdir = local["mkdir"]['-p'][res_dir_path]
        mkdir & FG
        h = hashlib.new('sha256')
        for trial in generate_trials(template, param_config):
            h.update(json.dumps(trial).encode('utf-8'))
            cur_trial_name = h.hexdigest()[-8:]
            with open(res_dir_path.joinpath(cur_trial_name + ".yaml"), 'w') as outf:
                yaml.dump(trial, outf, default_flow_style=False)
    else:
        relative_dir_path = Path(args.model_name).joinpath(args.warm_start)
        res_dir_path = storage_resolver['trials'].joinpath(relative_dir_path)
        if res_dir_path.name not in os.listdir(res_dir_path.parent):
            raise "Wrong timestamp for warm start"

    for exp in args.experiments:
        print(f"=====starting experiment {exp}=====")
        cur_outfile_name = res_dir_path.joinpath(f"{exp.replace('/', '_')}_finished_runs.txt")

        with open(cur_outfile_name, 'a+') as outfile:
            print(f'trials stored to {cur_outfile_name}')
            outfile.seek(0)
            already_run = set(outfile.read().split())
            outfile.seek(0, 2)

            for trial in tqdm(os.listdir(res_dir_path)):
                if trial.endswith(".yaml"):
                    trial = relative_dir_path.joinpath(trial[:-5])
                    if args.warm_start is None or str(trial) not in already_run:
                        run_exp = local["python"]["run_experiments.py"]['--experiments'][exp]['--trials'][trial] \
                            ['--gpus']['0']['--wandb-entity'][args.wandb_entity]
                        run_exp & FG
                        print(trial, file=outfile, end=" ", flush=True)
                    else:
                        print(f"restored predictions for trial {trial}")


if __name__ == '__main__':
    main()
