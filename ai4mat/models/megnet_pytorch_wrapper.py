import pandas as pd
from ai4mat.models.megnet_pytorch.megnet_pytorch_trainer import MEGNetPyTorchTrainer


def get_megnet_pytorch_predictions(
        train_structures: pd.Series,  # series of pymatgen object
        train_targets: pd.Series,  # series of scalars
        test_structures: pd.Series,  # series of pymatgen object
        test_targets: pd.Series,  # series of scalars
        target_is_intensive: bool,
        model_params: dict,
        gpu: int,
        checkpoint_path,
        use_last_checkpoint=True
        ):
    print(model_params)

    model = MEGNetPyTorchTrainer(
        train_structures,
        train_targets,
        test_structures,
        test_targets,
        configs=model_params,
        gpu_id=gpu,
        save_checkpoint=False,
    )
    model.train()

    print('========== predicting ==============')
    return model.predict_structures()
