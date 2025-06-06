import numpy as np
from sklearn.metrics import r2_score


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def R2(pred, true):
    return r2_score(pred, true)


def index_of_agreement(observed, predicted):
    observed = np.array(observed)
    predicted = np.array(predicted)

    mean_observed = np.mean(observed)

    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((np.abs(observed - mean_observed) + np.abs(predicted - mean_observed)) ** 2)

    ia = 1 - (numerator / denominator)

    return ia

def metric(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    ia = index_of_agreement(true, pred)

    return mae, rmse, ia