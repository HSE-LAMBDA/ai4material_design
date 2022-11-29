from typing import Optional
import logging
import pandas as pd
import numpy as np

from ai4mat.models.schnet.schnet_trainer import SchNetTrainer


def get_schnet_predictions(
        train_structures: pd.Series, # Structure
        train_targets: pd.Series, # float
        train_weights: Optional[pd.Series], # float
        test_structures: pd.Series, # Structure
        test_targets: pd.Series, # float
        test_weights: Optional[pd.Series], # float
        target_is_intensive: bool,
        model_params: dict,
        gpu: Optional[int],
        checkpoint_path: Optional[str],
        n_jobs: Optional[int],
        minority_class_upsampling: bool) -> np.ndarray:
    """
    Computes SchNet predictions.
    Args:
        train_structures: Training structures.
        train_targets: Training targets.
        train_weights: Training weights.
        test_structures: Test structures.
        test_targets: Test targets.
        test_weights: Test weights.
        target_is_intensive: Whether the target is intensive2.
        model_params: Model parameters.
        gpu: GPU ID. None for CPU.
        checkpoint_path: Checkpoint path, not used, but is retained for compatibility.
        n_jobs: Number of jobs, not used, but is retained for compatibility.
        minority_class_upsampling: Whether to use minority class upsampling, must be specified is weights are used.
    Returns:
        Predictions on the test set.
    """
    model_params_copy = model_params.copy()
    if "minority_class_upsampling" in model_params_copy['optim']:
        if model_params_copy['optim']['minority_class_upsampling'] != minority_class_upsampling:
            raise ValueError("Minority class upsampling does not match model")
    elif minority_class_upsampling is not None:
        model_params_copy['optim']['minority_class_upsampling'] = minority_class_upsampling
    if train_weights is not None and not minority_class_upsampling:
        raise ValueError("Train weights are only supported with minority class upsampling")
    if checkpoint_path is not None:
        logging.warning("SchNet doesn't use checkpoints")
    if n_jobs is not None:
        logging.warning("SchNet doesn't use n_jobs")

    if train_weights is not None:
        [setattr(s, 'weight', w) for s, w in zip(train_structures, train_weights)]
    if test_weights is not None:
        [setattr(s, "weight", w) for s, w in zip(test_structures, test_weights)]
    if "readout" in model_params_copy['model']:
        logging.warning("SchNet readout is handled per target based on target_is_intensive")
    if target_is_intensive:
        model_params_copy['model']["readout"] = "mean"
    else:
        model_params_copy['model']["readout"] = "add"
    model = SchNetTrainer(
        train_structures,
        train_targets,
        test_structures,
        test_targets,
        configs=model_params_copy,
        gpu_id=gpu,
        target_name=train_targets.name
    )
    print("=========== Training ===============")
    model.train()
    print("========== Predicting ==============")
    predictions = model.predict_structures(test_structures)
    return predictions.ravel()