import argparse
import yaml
from pathlib import Path
import pandas

import sys
sys.path.append(".")
from ai4mat.data.data import (
    StorageResolver,
    read_structures_descriptions,
    read_defects_descriptions,
    get_prediction_path
)

def main():
    parser = argparse.ArgumentParser("Combines predictions with defect IDs")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--trial", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    storage_resolver = StorageResolver()
    experiment_path = storage_resolver["experiments"].joinpath(args.experiment)
    with open(experiment_path.joinpath("config.yaml")) as experiment_file:
        experiment = yaml.safe_load(experiment_file)
    assert len(experiment["datasets"]) == 1
    structures_descriptions = read_structures_descriptions(Path(experiment["datasets"][0]))

    all_predictions = []
    for target in experiment["targets"]:
        all_predictions.append(pandas.read_csv(storage_resolver["predictions"].joinpath(
                get_prediction_path(args.experiment, target, args.trial)), index_col="_id"))
    predictions = pandas.concat(all_predictions, axis=1)
    
    assert structures_descriptions.index.equals(predictions.index)
    predictions["descriptor_id"] = structures_descriptions["descriptor_id"]
    assert(args.output_file.endswith(".csv"))
    predictions.to_csv(args.output_file, index_label="_id")

if __name__ == "__main__":
    main()