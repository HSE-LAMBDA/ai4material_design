#!/bin/bash
TRIALS_PATH="trials"
PREDICTIONS_PATH="datasets/predictions"
EXPERIMENT_NAME="combined_mixed_weighted_validation"
LOG_FOLDER=nscc_logs/$(date "+%F-%H-%M-%S")
TARGETS=("formation_energy_per_site" "homo_lumo_gap_min")
mkdir $LOG_FOLDER

rm missing_trials.txt
rm missing_targets.txt
for TRIAL in "$TRIALS_PATH"/megnet_pytorch/09-11-2022_18-11-54/*.yaml; do
    TRIAL_YAML=$(realpath --relative-to="$TRIALS_PATH" "$TRIAL")
    TRIAL_NAME=${TRIAL_YAML%.yaml}
    TRIALS_TO_PROCESS+=("$TRIAL_NAME")
    for TARGET in ${TARGETS[@]}; do
        if [ ! -f "$PREDICTIONS_PATH"/"$EXPERIMENT_NAME"/"$TARGET"/"$TRIAL_NAME".csv.gz ]; then
            echo $TRIAL_NAME >> missing_trials.txt
            echo $TARGET >> missing_targets.txt
            #JOB_NAME=${EXPERIMENT_NAME////-}_${TARGET}_${TRIAL_NAME////-}
            #echo qsub -o "$LOG_FOLDER/$JOB_NAME".oe -N $JOB_NAME -v EXPERIMENT="$EXPERIMENT_NAME",TRIALS="$TRIAL_NAME",TARGETS="$TARGET" scripts/run_experiments_nscc.pbs
        fi
    done    
done
# megnet_pytorch/09-11-2022_18-11-54/65bdf06b