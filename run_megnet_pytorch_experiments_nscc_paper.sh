#!/bin/bash
TRIALS_PATH="trials"
PREDICTIONS_PATH="datasets/predictions"
EXPERIMENT_NAME="combined_mixed_weighted_validation"
LOG_FOLDER=nscc_logs/$(date "+%F-%H-%M-%S")
mkdir $LOG_FOLDER
JOB_ID=0

launch_job() {
    echo "Running trials: ${1}"
    local JOB_NAME=ai4material_design_${EXPERIMENT_NAME////-}_megnet_sparse_${JOB_ID}
    JOB_ID=$((JOB_ID+1))
    qsub -o "$LOG_FOLDER/$JOB_NAME".oe -N $JOB_NAME -v EXPERIMENT="$EXPERIMENT_NAME",TRIALS="$1" scripts/run_experiments_nscc.pbs
}

declare -a TRIALS_TO_PROCESS
PROCESSED_TRIALS=0
BATCH_SIZE=4
for TRIAL in "$TRIALS_PATH"/megnet_pytorch/09-11-2022_18-11-54/*.yaml; do
    TRIAL_YAML=$(realpath --relative-to="$TRIALS_PATH" "$TRIAL")
    TRIAL_NAME=${TRIAL_YAML%.yaml}
    TRIALS_TO_PROCESS+=("$TRIAL_NAME")
    PROCESSED_TRIALS=$((PROCESSED_TRIALS+1))
    if [ $PROCESSED_TRIALS -eq $BATCH_SIZE ]; then
        TRIALS_STR="${TRIALS_TO_PROCESS[@]}"
        launch_job "$TRIALS_STR"
        TRIALS_TO_PROCESS=()
        PROCESSED_TRIALS=0
    fi
done
if [ $PROCESSED_TRIALS -gt 0 ]; then
    launch_job "$TRIALS_TO_PROCESS"
fi