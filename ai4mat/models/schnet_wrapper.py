import os

from ai4mat.models.schnet.schnet_trainer import SchNetTrainer


def get_schnet_predictions(
    train_structures,
    train_targets,
    test_structures,
    test_targets,
    target_is_intensive,
    model_params,
    gpu,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    model = SchNetTrainer(
        train_structures,
        train_targets,
        test_structures,
        test_targets,
        configs=model_params,
        save_checkpoint=False,
        gpu_id=gpu,
        verbose=1,
    )

    # We use the same test for monitoring, but do no early stopping
    print("=========== Training ===============")
    model.train()
    print("========== Predicting ==============")
    predictions = model.predict_structures(test_structures)
    return predictions.ravel()
