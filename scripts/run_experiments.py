import argparse
import pathlib
import yaml

from megnet_wrapper import get_megnet_predictions

def run(params_tuple):
    experiment_path, trial_path, gpu, test_fold = params_tuple
    with open(pathlib.Path(experiment_path, "config.yaml")) as experiment_config_file:
        experiment_config = yaml.load(experiment_config_file)
    with open(trial_path) as trial_file:
        trial_config = yaml.load(trial_file)

    

def main():
    parser = argparse.ArgumentParser("Runs experiments")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--trials", type=str, nargs="+")
    parser.add_argument("--predictions_root", type=str)
    parser.add_argument("--gpus", type=int, nargs="*")
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--processes-per-gpu", type=int, default=3)
    args = parser.parse_args()

    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_RUN_GROUP"] = "2D-crystal-" + wandb.util.generate_id()
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    with Pool(len(args.gpus)*args.proceses_per_gpu) as pool:
        for experiment in args.experiments:
            pool.map(run_on_gpu, enumerate(experiments))

