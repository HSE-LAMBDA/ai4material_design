#!/bin/bash
MODEL=$1
TRIALS_PACK=$2

if [ -z "$MODEL" -o -z "$TRIALS_PACK" ]; then
    echo "Usage: $0 MODEL TRIALS_PACK"
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

for TRIAL in "$TRIALS_PATH"/"$MODEL"/"$TRIALS_PACK"/*.yaml; do
    TRIAL_YAML=$(realpath --relative-to="$TRIALS_PATH" "$TRIAL")
    TRIAL_NAME=${TRIAL_YAML%.yaml}
    TRIALS_TO_PROCESS+=("$TRIAL_NAME")
    for TARGET in ${TARGETS[@]}; do
        if [ ! -f "$PREDICTIONS_PATH"/"$EXPERIMENT_NAME"/"$TARGET"/"$TRIAL_NAME".csv.gz ]; then
            echo CHANGE THIS TO RUN THE JOB IN THE WAY YOU WANT
            #python run_experiments.py --targets $TARGET --cpu --experiments $EXPERIMENT_NAME --trials $TRIAL_NAME --wandb-entity hse_lambda --n-jobs 40
            #JOB_NAME=${EXPERIMENT_NAME////-}_${TARGET}_${TRIAL_NAME////-}
            #qsub -o "$LOG_FOLDER/$JOB_NAME".oe -N $JOB_NAME -v EXPERIMENT="$EXPERIMENT_NAME",TRIALS="$TRIAL_NAME",TARGETS="$TAGRET" $SCRIPTPATH/run_experiments_nscc.pbs
        fi
    done    
done