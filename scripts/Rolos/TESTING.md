Hello! The general objective of the test is to make sure that a user with no knowledge of the project or Rolos can reproduce a meaningful part the project.
# Preparation
Remove the intermediate files. The actual user will have them, of course, but we want to make sure that the user can produce them with Rolos if needed.
```bash
cd ai4material_design
git rm -r datasets/csv_cif/low_density_defects_Innopolis-v1
git rm -r datasets/csv_cif/high_density_defects
git rm -r datasets/csv_cif/low_density_defects

git rm -r datasets/processed/high_density_defects
git rm -r datasets/processed/low_density_defects

git rm -r datasets/predictions/combined_mixed_weighted_test/*/stability/catboost

git commit -m "Clean up the intermediate files for testing"
git push
```
# Steps to reproduce
Follow [CONSTRUCTOR.md](../../docs/CONSTRUCTOR.md). In the interest of both yours and computational time, don't reproduce everything, but see the notes:
1. Data preprocessing: VASP -> csv/cif -> pickle & matminer - run only one of the matminer nodes, any one with high_density_defects
2. Run the experiment on train/test split - run the CatBoost workflow in the full, and any one node from each of the other workflows
3. Result analysis - full
# Troubleshooting
In case you have problems, please write to [Nikita](https://t.me/kazeevn)