import pandas as pd
import os


def set_y(structure, y):
    setattr(structure, "y", y)
    return structure


def get_megnet_pytorch_predictions(
        train_structures: pd.Series,  # series of pymatgen object
        train_targets: pd.Series,  # series of scalars
        test_structures: pd.Series,  # series of pymatgen object
        test_targets: pd.Series,  # series of scalars
        target_is_intensive: bool,
        model_params: dict,
        gpu: int,
        checkpoint_path,
        ):
    if not target_is_intensive:
        raise NotImplementedError
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    target_name = model_params['model']['target_name']
    train_targets = train_targets.tolist()
    test_targets = test_targets.tolist()

    train_data = [set_y(s, y) for s, y in zip(train_structures, train_targets)]
    test_data = [set_y(s, y) for s, y in zip(test_structures, test_targets)]

    model = MEGNetPyTorchTrainer(
        train_data,
        test_data,
        target_name,
        configs=model_params,
        gpu_id=gpu,
        save_checkpoint=False,
    )
    model.train()

    print('========== predicting ==============')
    #return model.predict_test_structures()
    return model