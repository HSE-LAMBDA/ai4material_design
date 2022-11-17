import argparse
import yaml
import hashlib
import json
import numpy as np
import sys

from itertools import product
from copy import deepcopy

from pathlib import Path
from datetime import datetime

sys.path.append('.')
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
                cur_set.extend([cur_path, value])
            sets.append(cur_set)
        else:
            dfs(value, path, sets, split)
        path.pop()


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
    dfs(param_config, [], param_sets, split=False)

    for step in range(n_steps):
        cur_template = deepcopy(template)
        for param in param_sets:
            if param[1][0] == 'float_min_max':
                cur_param_value = np.random.uniform(param[1][1], param[1][2])
            elif param[1][0] == 'int_min_max':
                cur_param_value = int(np.random.uniform(param[1][1], param[1][2]))
            elif param[1][0] == 'grid':
                cur_param_value = np.random.choice(param[1][1:]).item()
            else:
                raise ValueError('unknown distribution')
            set_item_by_path(cur_template, cur_param_value, list(param[0].split()))
        yield cur_template


def parse_args():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=['grid', 'random'], required=True)
    parser.add_argument('--n-steps', type=int)
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
    res_dir_path.mkdir(parents=True, exist_ok=True)
    h = hashlib.new('sha256')
    generator = generate_grid_trials(template, param_config) if args.mode == 'grid' else \
        generate_random_trials(template, param_config, args.n_steps)
    for trial in generator:
        h.update(json.dumps(trial).encode('utf-8'))
        cur_trial_name = h.hexdigest()[-8:]
        with open(res_dir_path.joinpath(cur_trial_name + ".yaml"), 'w') as outf:
            yaml.dump(trial, outf, default_flow_style=False)
    print(f'stored trials to {relative_dir_path}')


if __name__ == '__main__':
    main()