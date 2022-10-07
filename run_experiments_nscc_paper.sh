#!/bin/bash
EXPERIMENTS_PATH="datasets/experiments"
TRIALS_PATH="trials"
for EXPERIMENT in "$EXPERIMENTS_PATH"/high_density/*/ "$EXPERIMENTS_PATH"/low_density/*/ "$EXPERIMENTS_PATH"/low_high_combined; do
    if [ ! -d "$EXPERIMENT" ]; then
        echo "$EXPERIMENT does not exist."
        continue
    fi
    EXPERIMENT_NAME=$(realpath --relative-to="$EXPERIMENTS_PATH" "$EXPERIMENT")
    for TRIAL in "$TRIALS_PATH"/megnet_pytorch_paper/*.yaml; do
        TRIAL_YAML=$(realpath --relative-to="$TRIALS_PATH" "$TRIAL")
        TRIAL_NAME=${TRIAL_YAML%.yaml}
        qsub -N ai4material_design_${EXPERIMENT_NAME////-}_${TRIAL_NAME////-} -v EXPERIMENT="$EXPERIMENT_NAME",TRIAL="$TRIAL_NAME" scripts/run_experiments_nscc.pbs
    done
done
