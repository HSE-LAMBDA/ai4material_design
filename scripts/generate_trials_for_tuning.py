import argparse
import yaml
import hashlib
import json
import numpy as np

from itertools import product
from copy import deepcopy

from pathlib import Path
from plumbum import local, FG
from datetime import datetime

from ai4mat.data.data import StorageResolver


def dfs(data, path, sets, split=True):
    """
    walk through nested dict and store all paths in format 'a b c'
    """
    for key, value in data.items():
        path.append(key)
        if type(value) == list:
            cur_set = []
            cur_path = " ".join(path)
            if split:
                for val in value:
                    cur_set.append((cur_path, val))
            else:
                cur_set.append((cur_path, value))
            sets.append(cur_set)
        else:
            dfs(value, path, sets, split)
        path = path[:-1]


def set_item_by_path(d, value, path):
    if len(path) == 1:
        d[path[-1]] = value
    else:
        set_item_by_path(d[path[0]], value, path[1:])


def generate_grid_trials(template, param_config):
    param_sets = []
    dfs(param_config, [], param_sets)

    for params_change in product(*param_sets):
        cur_template = deepcopy(template)
        for param in params_change:
            set_item_by_path(cur_template, param[1], list(param[0].split()))
        yield cur_template


def generate_random_trials(template, param_config, n_steps):
    param_sets = []
    dfs(param_config, [], param_sets)

    for step in range(n_steps):
        cur_template = deepcopy(template)
        for param in param_sets:
            cur_param_value = np.random.uniform(param[1][0], param[1][1])
            if isinstance(param[1][0], int):
                cur_param_value = int(cur_param_value)
            set_item_by_path(cur_template, cur_param_value, list(param[0].split()))
        yield cur_template


def parse_args():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--mode", choices=['grid, random'], required=True)
    parser.add_argument('n_steps', type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'random' and args.n_steps is None:
        raise ValueError('please provide number of steps for random search')

    storage_resolver = StorageResolver()

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
    generator = generate_grid_trials(template, param_config) if args.mode == 'grid' else \
        generate_random_trials(template, param_config, args.n_steps)
    for trial in generator:
        h.update(json.dumps(trial).encode('utf-8'))
        cur_trial_name = h.hexdigest()[-8:]
        with open(res_dir_path.joinpath(cur_trial_name + ".yaml"), 'w') as outf:
            yaml.dump(trial, outf, default_flow_style=False)


if __name__ == '__main__':
    main()
