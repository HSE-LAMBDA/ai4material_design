from catboost import CatBoostRegressor
import pandas as pd


def get_catboost_predictions(
    x_train, y_train, x_test, y_test, target_is_intensive, model_params, gpu
):
    model = CatBoostRegressor(
        task_type="GPU", bootstrap_type="Bernoulli", devices=str(gpu), **model_params
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    predictions = pd.Series(predictions, index=x_test.index)
    return predictions
