#!/bin/bash
OUTPUT_FOLDER=$1
SCRIPT_FODLER="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
INPUT_FOLDER=$SCRIPT_FODLER/../..
TARGET="formation_energy_per_site"
for EXPERIMENT in MoS2_V2 combined_mixed_weighted_test; do
    PREDICTION_PATH=datasets/predictions/$EXPERIMENT/$TARGET
    while read trial; do
        mkdir -p $(dirname $OUTPUT_FOLDER/$PREDICTION_PATH/${trial}.csv.gz) 
        cp $INPUT_FOLDER/$PREDICTION_PATH/stability/${trial}/1.csv.gz $OUTPUT_FOLDER/$PREDICTION_PATH/${trial}.csv.gz
        echo $OUTPUT_FOLDER/$PREDICTION_PATH/${trial}.csv.gz
    done < ${SCRIPT_FODLER}/${EXPERIMENT}.txt
done

