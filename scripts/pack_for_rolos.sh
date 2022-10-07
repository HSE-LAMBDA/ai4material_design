#!/bin/bash

TARGET_DIR=datasets/others/rolos/2d-materials-point-defects
mkdir -p $TARGET_DIR
root=$(pwd)
for material in BP_spin GaSe_spin hBN_spin InSe_spin MoS2 WSe2
do
    archive_path=$(realpath $TARGET_DIR/${material}_high_concentration.zip)
    rm $archive_path
    cd datasets/csv_cif/high_density_defects/${material}_500/
    zip -r $archive_path ./*
    cd $root
    cd datasets/processed/high_density_defects/${material}_500/
    zip $archive_path targets.csv.gz
    cd $root
done
for material in MoS2 WSe2
do
    archive_path=$(realpath $TARGET_DIR/${material}_low_concentration.zip)
    rm $archive_path
    cd datasets/csv_cif/low_density_defects/${material}/
    zip -r $archive_path ./*
    cd $root
    cd datasets/processed/low_density_defects/${material}/
    zip $archive_path targets.csv.gz
    cd $root
done
cd $TARGET_DIR
rm 2d-materials-point-defects-all.zip
zip 2d-materials-point-defects-all.zip *.zip