Hello! The general objective of the test is to make sure that a user with no knowelage of the project or Rolos can reproduce the project.
# Preparation
Remove the intermidiate files. The actual user will have them, of course, but we want to make sure that the user can produce them with Rolos if needed.
```bash
git rm -r datasets/csv_cif/low_density_defects_Innopolis-v1
git rm -r datasets/csv_cif/high_density_defects
git rm -r datasets/csv_cif/low_density_defects

git rm -r datasets/processed/high_density_defects
git rm -r datasets/processed/low_density_defects

git rm -r datasets/predictions/combined_mixed_weighted_test/*/stability/catboost

git commit -m "Clean up the intermidiate files for testing"
git push
```
# Steps to reproduce
In general, follow the README. In the interest of both yours and computational time, don't reproduce everything, but see the notes:
1. Running the pilot NN model - full
2. Running a pilot CatBoost model - don't recompute `matminer.csv.gz`
3. Data preprocessing: VASP -> csv/cif -> pickle & matminer - run only one of the matminer nodes, any one with high_density_defects
4. Hyperparameter optimisation - skip
5. Run the experiment on train/test split - run the CatBoost workflow in the full, and any one node from the the other workflows
6. Ablation study - skip
7. Result analysis - full
# Troubleshooting
In case you can't figure something out in more than 5 minutes, please write to [Nikita](https://t.me/kazeevn)