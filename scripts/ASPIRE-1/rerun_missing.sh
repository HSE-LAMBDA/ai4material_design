#!/bin/bash
MODEL=$1
TRIALS_PACK=$2
BATCH_SIZE=$3
if [ -z "$MODEL" -o -z "$TRIALS_PACK" -o -z "$BATCH_SIZE" ]; then
    echo "Usage: $0 MODEL TRIALS_PACK BATCH_SIZE"
    exit 1
fi

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PROJECT_ROOT=$(realpath "$SCRIPTPATH/../../")

TRIALS_PATH=$PROJECT_ROOT/"trials"
PREDICTIONS_PATH="$PROJECT_ROOT/datasets/predictions"
EXPERIMENT_NAME="combined_mixed_weighted_validation"
TARGETS=("formation_energy_per_site" "homo_lumo_gap_min")

LOG_FOLDER=$PROJECT_ROOT/nscc_logs/$(date "+%F-%H-%M-%S")
mkdir $LOG_FOLDER
JOB_ID=0

launch_job() {
    echo "Running trials: ${1} for target ${2}"
    local JOB_NAME=ai4material_design_${EXPERIMENT_NAME////-}_${MODEL}_${JOB_ID}
    JOB_ID=$((JOB_ID+1))
    qsub -o "$LOG_FOLDER/$JOB_NAME".oe -N $JOB_NAME -v EXPERIMENT="$EXPERIMENT_NAME",TRIALS="$1",TARGETS="$2" $SCRIPTPATH/run_experiments_nscc.pbs
}

declare -a TRIALS_TO_PROCESS
PROCESSED_TRIALS=0
for TARGET in ${TARGETS[@]}; do
    for TRIAL in "$TRIALS_PATH"/"$MODEL"/"$TRIALS_PACK"/*.yaml; do
        TRIAL_YAML=$(realpath --relative-to="$TRIALS_PATH" "$TRIAL")
        TRIAL_NAME=${TRIAL_YAML%.yaml}
        if [ ! -f "$PREDICTIONS_PATH"/"$EXPERIMENT_NAME"/"$TARGET"/"$TRIAL_NAME".csv.gz ]; then
            TRIALS_TO_PROCESS+=("$TRIAL_NAME")
            PROCESSED_TRIALS=$((PROCESSED_TRIALS+1))
        fi
        if [ $PROCESSED_TRIALS -eq $BATCH_SIZE ]; then
            TRIALS_STR="${TRIALS_TO_PROCESS[@]}"
            launch_job "$TRIALS_STR" "$TARGET"
            TRIALS_TO_PROCESS=()
            PROCESSED_TRIALS=0
        fi
    done
    if [ $PROCESSED_TRIALS -gt 0 ]; then
        TRIALS_STR="${TRIALS_TO_PROCESS[@]}"
        launch_job "$TRIALS_STR" "$TARGET"
        TRIALS_TO_PROCESS=()
        PROCESSED_TRIALS=0    
    fi  
done
