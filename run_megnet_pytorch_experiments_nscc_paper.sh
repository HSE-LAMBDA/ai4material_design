#!/bin/bash
TRIALS_PATH="trials"
EXPERIMENT_NAME="combined_mixed_weighted_validation"
LOG_FOLDER=nscc_logs/$(date "+%F-%H-%M-%S")
mkdir $LOG_FOLDER
for TRIAL in "$TRIALS_PATH"/megnet_pytorch/09-11-2022_18-11-54/*.yaml; do
    TRIAL_YAML=$(realpath --relative-to="$TRIALS_PATH" "$TRIAL")
    TRIAL_NAME=${TRIAL_YAML%.yaml}
    JOB_NAME=ai4material_design_${EXPERIMENT_NAME////-}_${TRIAL_NAME////-}
    qsub -o "$LOG_FOLDER/$JOB_NAME" -N $JOB_NAME -v EXPERIMENT="$EXPERIMENT_NAME",TRIAL="$TRIAL_NAME" scripts/run_experiments_nscc.pbs
done