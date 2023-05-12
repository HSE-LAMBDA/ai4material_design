#!/bin/bash
TRIAL_NAME=$1
TARGET=$2
REPEATS=$3
BATCH_SIZE=$4
EXPERIMENT_NAME=$5
TOTAL_CPU_COUNT=8

if [ -z "$TRIAL_NAME" -o -z "$TARGET" -o -z "$REPEATS" -o -z "$BATCH_SIZE" -o -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 TRIAL_NAME TARGET REPEATS BATCH_SIZE EXPERIMENT_NAME"
    exit 1
fi

if [ $((REPEATS % BATCH_SIZE)) -ne 0 ]; then
    echo "REPEATS must be a multiple of BATCH_SIZE"
    exit 1
fi

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PROJECT_ROOT=$(realpath "$SCRIPTPATH/../../")

TRIALS_PATH=$PROJECT_ROOT/"trials"
mkdir -p $TRIALS_PATH/stability/$TRIAL_NAME
THIS_WORKFLOW_PATH=$SCRIPTPATH/workflows/run_experiments/$EXPERIMENT_NAME/$TARGET/$TRIAL_NAME
mkdir -p $THIS_WORKFLOW_PATH
for i in $(seq 1 $REPEATS); do
    # Since for CatBoost we manually change the random seed, don't overwrite the present files
    if [ ! -f $TRIALS_PATH/stability/$TRIAL_NAME/$i.yaml ]; then
        cp $TRIALS_PATH/$TRIAL_NAME.yaml $TRIALS_PATH/stability/$TRIAL_NAME/$i.yaml
    fi
    # Check if we have a batch of trials to process
    if [ $((i % BATCH_SIZE)) -eq 0 ]; then
        TRIALS_BATCH=$(seq -f stability/$TRIAL_NAME/%g $((i - BATCH_SIZE + 1)) $i)
        THIS_SCRIPT_PATH=$THIS_WORKFLOW_PATH/node_$((i / BATCH_SIZE)).sh
        echo "#!/bin/bash" > $THIS_SCRIPT_PATH
        echo "cd ai4material_design" >> $THIS_SCRIPT_PATH
        echo "source scripts/Rolos/wandb_config.sh" >> $THIS_SCRIPT_PATH
        if [ $BATCH_SIZE -gt 1 ]; then
            echo "parallel -j ${BATCH_SIZE} python run_experiments.py --n-jobs $((TOTAL_CPU_COUNT / BATCH_SIZE)) --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets ${TARGET} --experiments ${EXPERIMENT_NAME} --trials ::: ${TRIALS_BATCH//$'\n'/ }" >> $THIS_SCRIPT_PATH
        else
            echo "python run_experiments.py --n-jobs ${TOTAL_CPU_COUNT} --output-folder /output --gpus 0 --processes-per-unit 1 --wandb-entity hse_lambda --targets ${TARGET} --experiments ${EXPERIMENT_NAME} --trials ${TRIALS_BATCH//$'\n'/ }" >> $THIS_SCRIPT_PATH
        fi
        
    fi
done