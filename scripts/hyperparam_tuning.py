import argparse
import yaml
import os
import sys

from pathlib import Path
from plumbum import local, FG
from tqdm import tqdm

sys.path.append('.')
from ai4mat.data.data import StorageResolver


def parse_args():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--warm-start", action='store_true')
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--trials-folder", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    storage_resolver = StorageResolver()

    relative_dir_path = Path(args.model_name).joinpath(args.trials_folder)
    res_dir_path = storage_resolver['trials'].joinpath(relative_dir_path)
    if res_dir_path.name not in os.listdir(res_dir_path.parent):
        raise ValueError("Wrong timestamp for warm start")

    print(f"=====starting experiment {args.experiment}=====")

    exp_file_path = storage_resolver['experiments'].joinpath(args.experiment + '/config.yaml')
    with open(exp_file_path, 'r') as exp_file:
        exp = yaml.safe_load(exp_file)
    targets = exp['targets']

    prediction_folder = storage_resolver['predictions'].joinpath(args.experiment)

    for trial in tqdm(os.listdir(res_dir_path)):
        trial = relative_dir_path.joinpath(trial[:-5])
        need_to_restart = False
        if args.warm_start:
            for target in targets:
                cur_path = prediction_folder.joinpath(target).joinpath(str(trial) + '.csv.gz')
                if cur_path.parent.is_dir() and os.path.basename(cur_path) in os.listdir(cur_path.parent):
                    print(f'find target {target} for trial {trial}')
                else:
                    need_to_restart = True
                    print(f'not find target {target} for trial {trial}')
                    break
        else:
            need_to_restart = True

        if need_to_restart:
            run_exp = local["python"]["run_experiments.py"]['--experiments'][args.experiment]['--trials'][trial] \
                ['--gpus']['0']['--wandb-entity'][args.wandb_entity]
            run_exp & FG
        else:
            print(f'restored predictions for trial {trial}')


if __name__ == '__main__':
    main()
