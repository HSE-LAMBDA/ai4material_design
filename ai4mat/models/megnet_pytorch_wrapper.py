import pandas as pd
from ai4mat.models.megnet_pytorch.megnet_pytorch_trainer import MEGNetPyTorchTrainer
from ai4mat.models.megnet_pytorch.struct2graph import SimpleCrystalConverter, FlattenGaussianDistanceConverter


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
        use_last_checkpoint=True
        ):
    train_targets = train_targets.tolist()
    test_targets = test_targets.tolist()

    train_data = [set_y(s, y) for s, y in zip(train_structures, train_targets)]
    test_data = [set_y(s, y) for s, y in zip(test_structures, test_targets)]

    # c = SimpleCrystalConverter(bond_converter=FlattenGaussianDistanceConverter(), add_z_bond_coord=True, cutoff=0.5)
    # s = c.convert(train_data[0])
    # print(s)
    # return

    model = MEGNetPyTorchTrainer(
        train_data,
        test_data,
        configs=model_params,
        gpu_id=gpu,
        save_checkpoint=False,
    )
    model.train()

    print('========== predicting ==============')
    return model.predict_test_structures()
