"""Benchmark problems for the BO method search (ORACLE -- do not edit during method iteration)."""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems.single import Ackley, Griewank, Rastrigin, Rosenbrock, Sphere, Zakharov


class RotatedEllipsoid(Problem):
    """Ill-conditioned, non-separable quadratic: the clean test of metric/rotation learning.

    ``f(x) = sum_i cond^(i/(d-1)) * z_i^2`` with ``z = R (x - x*)`` for a fixed random rotation
    ``R``. Unlike Rosenbrock it has a *straight* (not curved) valley, so it isolates the
    ill-conditioning + rotation difficulty without the curved-valley confound -- the pure test of
    whether a method needs a Mahalanobis/PCA-rotated kernel. Optimum ``F*=0`` at ``x*``.

    Args:
        n_var: Dimensionality.
        cond: Condition number (ratio of largest to smallest axis stiffness).
        seed: Seed for the fixed rotation and optimum location.
    """

    def __init__(self, n_var=2, cond=1e6, seed=0):
        super().__init__(n_var=n_var, n_obj=1, xl=-5.0, xu=5.0)
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((n_var, n_var)))
        self.R = q
        # optimum placed off-center but well inside the box so it is not trivially at a corner/center
        self.x_opt = rng.uniform(-2.0, 2.0, size=n_var)
        exps = np.arange(n_var) / max(n_var - 1, 1)
        self.coef = cond**exps

    def _evaluate(self, x, out, *args, **kwargs):
        z = (x - self.x_opt) @ self.R.T
        out["F"] = (z**2 * self.coef).sum(axis=1, keepdims=True)


def benchmark_problems(dims=(2, 10)):
    """Return the fixed benchmark suite as ``[(name, problem), ...]`` (optimum F*=0 for all).

    Four functions spanning the difficulty axes -- Sphere (baseline), Rosenbrock (curved-valley
    local), Rastrigin (multimodal-funnel global), RotatedEllipsoid (conditioning + rotation) --
    each at the requested dimensionalities.

    Args:
        dims: Dimensionalities to instantiate each function at.

    Returns:
        List of ``(name, pymoo Problem)`` tuples.
    """
    out = []
    for d in dims:
        out.append((f"sphere_{d}d", Sphere(n_var=d)))
        out.append((f"rosenbrock_{d}d", Rosenbrock(n_var=d)))
        out.append((f"rastrigin_{d}d", Rastrigin(n_var=d)))
        out.append((f"rot_ellipsoid_{d}d", RotatedEllipsoid(n_var=d, seed=d)))
    return out


def heldout_problems(dims=(2, 10)):
    """Return a held-out validation suite (functions NOT tuned against) to catch overfitting.

    The method search optimizes the four ``benchmark_problems``; a method can quietly *overfit* to
    them. These three functions (Ackley, Griewank, Zakharov) are never tuned against -- a real
    improvement must hold here too. (Lesson learned the hard way: a benchmark-only +12% ECDF gain
    evaporated on held-out functions; only changes that improve both are kept.)

    Args:
        dims: Dimensionalities to instantiate each function at.

    Returns:
        List of ``(name, pymoo Problem)`` tuples.
    """
    out = []
    for d in dims:
        out.append((f"ackley_{d}d", Ackley(n_var=d)))
        out.append((f"griewank_{d}d", Griewank(n_var=d)))
        out.append((f"zakharov_{d}d", Zakharov(n_var=d)))
    return out
