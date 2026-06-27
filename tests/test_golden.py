"""Golden behavior-regression baselines for pysamoo's deterministic kernels.

These lock the numerical building blocks a refactor is most likely to silently
break: surrogate-accuracy indicators and total-constraint-violation aggregation.

Note on scope: full surrogate-assisted *runs* are intentionally not baselined
here. Their model-selection step is not reproducible run-to-run (it depends on
the global RNG / numerical tie-breaking inside the model pool), so a whole-run
golden value would be flaky. We baseline the deterministic math instead; see
docs/PERFORMANCE.md.
"""

import numpy as np
import pytest

from pysamoo.core.indicator import calc_mae, calc_mse, calc_r2, calc_rmse, calc_sign_error, kendall_tau
from pysamoo.core.tcv import TotalConstraintViolation


def _fixed_predictions():
    rng = np.random.RandomState(0)
    y_true = rng.rand(50)
    y_hat = y_true + 0.1 * rng.standard_normal(50)
    return y_true, y_hat


@pytest.mark.golden
def test_golden_indicators():
    """Surrogate-accuracy indicators on fixed predictions."""
    y_true, y_hat = _fixed_predictions()
    return {
        "mse": float(calc_mse(y_true, y_hat)),
        "rmse": float(calc_rmse(y_true, y_hat)),
        "mae": float(calc_mae(y_true, y_hat)),
        "r2": float(calc_r2(y_true, y_hat, trn_y=y_true)),
        "kendall_tau": float(kendall_tau(y_true, y_hat, trn_y=y_true)),
        "sign_error": float(calc_sign_error(y_true, y_hat)),
    }


@pytest.mark.golden
def test_golden_total_constraint_violation():
    """Total constraint violation aggregation on fixed constraint matrices."""
    rng = np.random.RandomState(1)
    G = rng.standard_normal((8, 3))  # inequality constraints (violation where > 0)
    H = rng.standard_normal((8, 2))  # equality constraints

    tcv = TotalConstraintViolation()
    cv = tcv.calc(G=G, H=H)
    return np.asarray(cv, dtype=float).ravel().tolist()
