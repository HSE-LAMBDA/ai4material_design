from ai4mat.models.megnet_wrapper import get_megnet_predictions
from ai4mat.models.gemnet_wrapper import get_gemnet_predictions

def get_predictor_by_name(name):
    if name == "megnet":
        return get_megnet_predictions
    if name == "gemnet":
        return get_gemnet_predictions
    else:
        raise ValueError("Unknown model name")