#!/bin/bash
TRIAL_NAME=$1
TARGET=$2
REPEATS=$3
BATCH_SIZE=$4
EXPERIMENT_NAME="combined_mixed_weighted_test"

if [ $((REPEATS % BATCH_SIZE)) -ne 0 ]; then
    echo "REPEATS must be a multiple of BATCH_SIZE"
    exit 1
fi

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PROJECT_ROOT=$(realpath "$SCRIPTPATH/../../")

TRIALS_PATH=$PROJECT_ROOT/"trials"
mkdir -p $TRIALS_PATH/stability/$TRIAL_NAME
for i in $(seq 1 $REPEATS); do
    cp $TRIALS_PATH/$TRIAL_NAME.yaml $TRIALS_PATH/stability/$TRIAL_NAME/$i.yaml
    # Check if we have a batch of trials to process
    if [ $((i % BATCH_SIZE)) -eq 0 ]; then
        TRIALS_BATCH=$(seq -f stability/$TRIAL_NAME/%g $((i - BATCH_SIZE + 1)) $i)
        qsub -v EXPERIMENT="$EXPERIMENT_NAME",TRIALS="$TRIALS_BATCH",TARGETS="$TARGET" $SCRIPTPATH/run_experiments_nscc.pbs
    fi
done
