import pandas as pd

from defect_representation import VacancyAwareStructureGraph, FlattenGaussianDistance
from data import get_pickle_path

def get_megnet_predictions(
        trial,
        datasets_paths,
        n_folds,
        folds,
        target,
        test_fold=None):
    data = pd.concat([pd.read_pickle(get_pickle_path(path))
                      for path in datasets_paths]],
                     axis=0)
    if test_fold is None:
        raise ValueError("Must have a test fold")
    if trial.representation != "sparse":
        raise NotImplementedError("Curently, only sparse representation is supported")

    train_folds = set(range(experiment.n_folds)) - set((test_fold,))
    train_ids = folds[folds.isin(train_folds)]
    train = data.reindex(index=train_ids.index)

    test_ids = folds[folds == test_fold]
    test = data.reindex(index=test_ids.index)

    run = wandb.init(project='ai4material_design',
                     entity=os.environ["WANDB_ENTITY"],
                     config={"trial": trial.__dict__,
                             "experiment": experiment.__dict__,
                             "test fold": test_fold},
                     reinit=True)
    nfeat_edge_per_dim = 10
    cutoff = 15
    if trial.add_bond_z_coord:
        bond_converter = FlattenGaussianDistance(
            np.linspace(0, cutoff, nfeat_edge_per_dim), 0.5)
    else:
        bond_converter = GaussianDistance(
            np.linspace(0, cutoff, nfeat_edge_per_dim), 0.5)
    graph_converter = VacancyAwareStructureGraph(
        bond_converter=bond_converter,
        atom_features=trial.atom_features,
        add_bond_z_coord=trial.add_bond_z_coord,
        cutoff=cutoff)
    # TODO(kazeevn) do we need to have a separate scaler for each
    # supercell replication?
    # In the intensive case, we seem fine anyway
    scaler = StandardScaler.from_training_data(train.defect_representation,
                                               train[target],
                                               is_intensive=IS_INTENSIVE[target])
    model = MEGNetModel(nfeat_edge=graph_converter.nfeat_edge*nfeat_edge_per_dim,
                        nfeat_node=graph_converter.nfeat_node,
                        nfeat_global=2,
                        graph_converter=graph_converter,
                        npass=2,
                        target_scaler=scaler,
                        metrics=["mae"],
                        lr=trial.learning_rate)
    if self.supercell_replication:
        model = train_with_supercell_replication(
            model,
            train.defect_representation,
            train[target],
            test.defect_representation,
            test[self.target],
            is_intensive=IS_INTENSIVE[self.target],
            callbacks=[wandb.keras.WandbCallback(save_model=False)],
            **self.supercell_replication)
    else:
        # We use the same test for monitoring, but do no early stopping
        model.train(train.defect_representation,
                    train[self.target],
                    test.defect_representation,
                    test[self.target],
                    epochs=trial.epochs,
                    callbacks=[wandb.keras.WandbCallback(save_model=False)],
                    save_checkpoint=False,
                    verbose=1)
    predictions = model.predict_structures(data.defect_representation)
    run.finish()
    return predictions
