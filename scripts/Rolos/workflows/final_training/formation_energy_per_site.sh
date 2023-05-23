#!/bin/bash
# Trains the sparse MegNet model on all the data, saves the weights
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
./scripts/final_training/train_final_energy.sh
fi