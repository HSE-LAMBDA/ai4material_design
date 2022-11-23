#!/bin/bash

#DATASETS=( high_density_defects/BP_spin_500 high_density_defects/GaSe_spin_500 high_density_defects/hBN_spin_500 high_density_defects/InSe_spin_500 high_density_defects/MoS2_500 high_density_defects/WSe2_500 low_density_defects/MoS2 low_density_defects/WSe2 pilot )
DATASETS=( low_density_defects/MoS2 low_density_defects/WSe2 )

for DATASET in "${DATASETS[@]}"; do
    qsub -v DATASET="$DATASET" matminer-48h.pbs
done