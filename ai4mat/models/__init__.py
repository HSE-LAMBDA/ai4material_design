#from ai4mat.models.megnet_wrapper import get_megnet_predictions
#from ai4mat.models.gemnet_wrapper import get_gemnet_predictions
#from ai4mat.models.schnet_wrapper import get_schnet_predictions
from ai4mat.models.catboost_wrapper import get_catboost_predictions
from ai4mat.models.megnet_pytorch_wrapper import get_megnet_pytorch_predictions


def get_predictor_by_name(name):
    if name == "megnet":
        return get_megnet_predictions
    elif name == "gemnet":
        return get_gemnet_predictions
    elif name == "schnet":
        return get_schnet_predictions
    elif name == "catboost":
        return get_catboost_predictions
    elif name == "megnet_pytorch":
        return get_megnet_pytorch_predictions
    else:
        raise ValueError("Unknown model name")
