#!/bin/bash

TARGET_DIR=datasets/others/rolos/2d-materials-point-defects
mkdir -p $TARGET_DIR
COMBINED_ARCHIVE=$(realpath $TARGET_DIR/2d-materials-point-defects-all.zip)
rm $COMBINED_ARCHIVE
root=$(pwd)
for material in BP_spin GaSe_spin hBN_spin InSe_spin MoS2 WSe2
do
    cd datasets/csv_cif/
    zip -r $COMBINED_ARCHIVE ./high_density_defects/${material}_500/
    cd $root
    cd datasets/processed/
    zip $COMBINED_ARCHIVE ./high_density_defects/${material}_500/targets.csv.gz
    cd $root
done
for material in MoS2 WSe2
do
    cd datasets/csv_cif/
    zip -r $COMBINED_ARCHIVE ./low_density_defects/${material}/
    cd $root
    cd datasets/processed/
    zip -r $COMBINED_ARCHIVE ./low_density_defects/${material}/targets.csv.gz
    cd $root
done