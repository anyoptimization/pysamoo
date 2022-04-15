import numpy as np
from scipy.stats import rankdata


def calc_mse(y_true, y_hat, **kwargs):
    return ((y_true - y_hat) ** 2).mean()


def calc_rmse(y_true, y_hat, **kwargs):
    return calc_mse(y_true, y_hat, **kwargs) ** 0.5


def calc_mae(y_true, y_hat, **kwargs):
    return np.abs(y_true - y_hat).mean()


def kendall_tau(y_true, y_hat, trn_y=None, **kwargs):
    assert trn_y is not None, "For kendall tau the ranking needs to be calculated which requires the training data!"

    a = rankdata(np.concatenate([trn_y, y_true]), method='min')
    b = rankdata(np.concatenate([trn_y, y_hat]), method='min')

    n = len(a)

    i, j = np.meshgrid(np.arange(n), np.arange(n))

    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]),
                                np.logical_and(a[i] > a[j], b[i] < b[j])).sum()

    # ndisordered = ndisordered / (n * (n - 1))

    return ndisordered


def calc_r2(y_true, y_hat, trn_y=None, **kwargs):
    if trn_y is None:
        trn_y = y_true

    return 1 - (calc_mse(y_true, y_hat) / calc_mse(trn_y, trn_y.mean()))


def calc_sign_error(y_true, y_hat, **kwargs):
    return (np.sign(y_true) != np.sign(y_hat)).mean()


INDICATORS = {
    "rmse": dict(sign=+1, func=calc_rmse),
    "mse": dict(sign=+1, func=calc_mse),
    "mae": dict(sign=+1, func=calc_mae),
    "kendall_tau": dict(sign=+1, func=kendall_tau),
    "r2": dict(sign=-1, func=calc_r2),
    "sign": dict(sign=-1, func=calc_sign_error),
}
