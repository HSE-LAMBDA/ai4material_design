import os
import numpy as np
import wandb.keras
from megnet.utils.preprocessing import StandardScaler
from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from typing import Dict
from ai4mat.common.defect_representation import VacancyAwareStructureGraph, FlattenGaussianDistance


def get_megnet_predictions(
        train_structures, # series of pymatgen object
        train_targets, # series of scalars
        test_structures, # series of pymatgen object
        test_targets, # series of scalars
        target_is_intensive: bool,
        model_params: Dict,
        gpu: int):
    # TODO(kazeevn) elegant device configration
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if model_params["add_bond_z_coord"]:
        bond_converter = FlattenGaussianDistance(
            np.linspace(0, model_params["cutoff"],
                        model_params["nfeat_edge_per_dim"]), 0.5)
    else:
        bond_converter = GaussianDistance(
            np.linspace(0, model_params["cutoff"],
                        model_params["nfeat_edge_per_dim"]), 0.5)
    graph_converter = VacancyAwareStructureGraph(
        bond_converter=bond_converter,
        atom_features=model_params["atom_features"],
        add_bond_z_coord=model_params["add_bond_z_coord"],
        cutoff=model_params["cutoff"])
    # TODO(kazeevn) do we need to have a separate scaler for each
    # supercell replication?
    # In the intensive case, we seem fine anyway
    scaler = StandardScaler.from_training_data(train_structures,
                                               train_targets,
                                               is_intensive=target_is_intensive)
    model = MEGNetModel(nfeat_edge=graph_converter.nfeat_edge*model_params["nfeat_edge_per_dim"],
                        nfeat_node=graph_converter.nfeat_node,
                        nfeat_global=2,
                        graph_converter=graph_converter,
                        npass=2,
                        target_scaler=scaler,
                        metrics=["mae"],
                        lr=model_params["learning_rate"])
    if "supercell_replication" in model_params:
        model = train_with_supercell_replication(
            model,
            train_structures,
            train_targets,
            test_structures,
            test_targets,
            is_intensive=target_is_intensive,
            callbacks=[wandb.keras.WandbCallback(save_model=False)],
            **model_params["supercell_replication"])
    else:
        # We use the same test for monitoring, but do no early stopping
        model.train(train_structures,
                    train_targets,
                    test_structures,
                    test_targets,
                    epochs=model_params["epochs"],
                    callbacks=[wandb.keras.WandbCallback(save_model=False)],
                    save_checkpoint=False,
                    verbose=1)
    predictions = model.predict_structures(test_structures)
    return predictions.ravel()


def train_with_supercell_replication(
        model, train_structures, train_target,
        test_structures, test_target,
        callbacks, is_intensive,
        epochs_per_replication_variant,
        replication_iterations,
        max_replications,
        random_seed):

    assert max_replications >= 1
    rng = np.random.RandomState(random_seed)

    for replication_iteration in range(replication_iterations):
        augmented_train = []
        replications = rng.randint(low=1,
                                   high=max_replications + 1,
                                   size=[len(train_structures), 2])
        for structure, replication_params in zip(train_structures, replications):
            augmented_train.append(structure.copy())
            augmented_train[-1].make_supercell([
                replication_params[0],
                replication_params[1],
                1])
            augmented_train[-1].state = structure.state
        if is_intensive:
            augmented_train_target = train_target
        else:
            augmented_train_target = train_target * replication_params.prod(axis=1)
        model.train(
            augmented_train,
            augmented_train_target,
            test_structures,
            test_target,
            epochs=epochs_per_replication_variant,
            callbacks=callbacks,
            save_checkpoint=False,
            verbose=True
        )
    return model
