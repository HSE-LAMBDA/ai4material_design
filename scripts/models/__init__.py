from .megnet_wrapper import get_megnet_predictions

def get_predictor_by_name(name):
    if name == "megnet":
        return get_megnet_predictions
    else:
        raise ValueError("Unknown model name")
