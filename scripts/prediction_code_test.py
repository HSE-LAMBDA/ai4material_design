# This is not really a script, but a code snippet to test the prediction code in the notebook
from ai4mat.data.data import read_experiment_datasets
experiment, folds, datasets, defects = read_experiment_datasets("combined_mixed_weighted_test")
predicted = pd.DataFrame(columns=list(predictors.keys()), index=datasets.index)
for target, predictor in predictors.items():
    predicted[target] = predictor.predict_structures(datasets.defect_representation)
    
predicted_pipeline = []
for target, trial_name in model_names.items():
    predicted_pipeline.append(pd.read_csv(StorageResolver()["predictions"] / "combined_mixed_weighted_test" / target / f"{trial_name}.csv.gz", index_col="_id"))
predicted_pipeline = pd.concat(predicted_pipeline, axis=1)
predicted_test_only = predicted.reindex(predicted_pipeline.index)

for target in model_names.keys():
    print((predicted_test_only[target] - predicted_pipeline[f"predicted_{target}_test"]).abs().mean())