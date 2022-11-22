import logging
import pandas as pd
from catboost import CatBoostRegressor, Pool
from wandb.catboost import WandbCallback, log_summary

def get_catboost_predictions(
    x_train, y_train, weights_train,
    x_test, y_test, weight_test,
    target_is_intensive,
    model_params,
    gpu,
    checkpoint_path,
    n_jobs=None,
    minority_class_upsampling=False):
    specific_params = model_params.copy()
    if checkpoint_path is not None:
        logging.warning("Checkpoint path is not supported for CatBoost")
    if minority_class_upsampling:
        raise NotImplemented("minority_class_upsampling makes isn't implemented for CatBoost")
    if gpu is None:
        specific_params["task_type"] = "CPU"
        specific_params["thread_count"] = n_jobs
    else:
        if n_jobs is not None:
            logging.warning("n_jobs isn't used when running on GPU")
        specific_params["task_type"] = "GPU"
        specific_params["devices"] = str(gpu)
    specific_params["eval_metric"] = "MAE"
    model = CatBoostRegressor(**specific_params)
    eval_pool = Pool(data=x_test, label=y_test, weight=weight_test)
    if gpu is None:
        callbacks = [WandbCallback()]
    else:
        callbacks = None
    model.fit(x_train,
              y_train,
              sample_weight=weights_train,
              eval_set=eval_pool,
              callbacks=callbacks,
              use_best_model=False)
    log_summary(model, save_model_checkpoint=False)
    predictions = model.predict(x_test)
    return predictions.reshape(-1, 1)
