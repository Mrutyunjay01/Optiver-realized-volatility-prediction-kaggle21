import numpy as np


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def feval_rmspe(y_pred, model, is_xgb=True):
    y_true = model.get_label()

    if is_xgb:
        return "RMSPE", rmspe(y_true, y_pred)

    return "RMSPE", rmspe(y_true, y_pred), False

