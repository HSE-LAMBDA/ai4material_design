import os

from ai4mat.models.gemnet.gemnet_trainer import GemNetTrainer


def get_gemnet_predictions(
        train_structures,
        train_targets,
        test_structures,
        test_targets,
        target_is_intensive,
        model_params,
        gpu,
        checkpoint_path,
        n_jobs=1,
        **kwargs
        ):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    model_params['model'].update({'extensive': target_is_intensive})
    model = GemNetTrainer(
        train_structures,
        train_targets,
        test_structures,
        test_targets,
        configs=model_params,
        save_checkpoint=True,
        checkpoint_path=checkpoint_path,
        gpu_id=gpu,
        verbose=1,
        )
   
        # We use the same test for monitoring, but do no early stopping
    model.train()
    print('========== predicting ==============')
    predictions = model.predict_structures(test_structures)
    return predictions