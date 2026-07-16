"""Shared machinery for the EGO-style surrogate-assisted algorithms (fit, predict, candidate pools)."""

from copy import deepcopy

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pysurrogate.dace import Exponential
from pysurrogate.models import Kriging


def default_kriging():
    """The default per-objective surrogate prototype (Kriging with an exponential/Matern-1/2 kernel).

    Returns:
        A fresh, unfitted ``Kriging`` model to be deep-copied per objective.
    """
    return Kriging(corr=Exponential())


def fit_per_objective(proto, X, F):
    """Deep-copy the prototype once per objective and fit each copy on that objective's column.

    Args:
        proto: The surrogate prototype to clone.
        X: Decision matrix, shape ``(n, n_var)``.
        F: Objective matrix, shape ``(n, n_obj)``.

    Returns:
        A list of fitted models, one per objective column of ``F``.
    """
    models = [deepcopy(proto) for _ in range(F.shape[1])]
    for m, model in enumerate(models):
        model.fit(X, F[:, m])
    return models


def predict_mu_sigma(models, X):
    """Predict the mean and standard deviation of each model at ``X`` in a single pass per model.

    ``predict(X, var=True)`` populates both the mean (``.y``) and the standard deviation (``.sigma``),
    so one call per model suffices (the earlier code predicted each candidate twice).

    Args:
        models: The fitted per-objective models.
        X: The points to predict at, shape ``(n, n_var)``.

    Returns:
        A tuple ``(mu, sigma)``, each of shape ``(n, len(models))``.
    """
    preds = [model.predict(X, var=True) for model in models]
    mu = np.column_stack([p.y[:, 0] for p in preds])
    sigma = np.column_stack([p.sigma[:, 0] for p in preds])
    return mu, sigma


def lhs_local_pool(problem, elite_X, n_pool, rng, scale=0.05):
    """A candidate pool: a space-filling LHS plus Gaussian perturbations of the elite designs.

    A pure LHS pool is too sparse in higher dimensions to locate the acquisition optimum, so half the
    pool is drawn near the current best (elite) designs. The RNG is drawn strictly in the order
    LHS -> integers -> standard_normal so results are reproducible for a fixed seed.

    Args:
        problem: The problem (for its box bounds and dimensionality).
        elite_X: The elite designs to perturb around, shape ``(k, n_var)``.
        n_pool: Size of each half of the pool (LHS and local); the result has ``2 * n_pool`` rows.
        rng: The run's numpy ``Generator``.
        scale: Perturbation width as a fraction of the box width.

    Returns:
        The candidate decision matrix, shape ``(2 * n_pool, n_var)``.
    """
    xl, xu = problem.xl, problem.xu
    cand = LHS().do(problem, n_pool, random_state=rng).get("X")
    idx = rng.integers(len(elite_X), size=n_pool)
    local = np.clip(elite_X[idx] + scale * (xu - xl) * rng.standard_normal((n_pool, problem.n_var)), xl, xu)
    return np.vstack([cand, local])


def front_and_hv(F, margin=0.1, eps=1e-9):
    """The current non-dominated front and a hypervolume indicator referenced past its nadir.

    Args:
        F: Objective matrix, shape ``(n, n_obj)``.
        margin: Fraction of the objective range added past the nadir for the reference point.
        eps: Floor on the per-objective range (guards against a degenerate zero-width front).

    Returns:
        A tuple ``(nds, front, hv)``: the non-dominated indices, the front ``F[nds]``, and a
        ``pymoo`` ``HV`` indicator with the reference point already set.
    """
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    front = F[nds]
    z_min, z_max = F.min(axis=0), F.max(axis=0)
    ref = z_max + margin * np.maximum(z_max - z_min, eps)
    return nds, front, HV(ref_point=ref)


def pareto_optimum(archive):
    """The non-dominated subset of an archive (the reported optimum for multi-objective methods).

    Args:
        archive: The evaluated population.

    Returns:
        The non-dominated members of ``archive``.
    """
    nds = NonDominatedSorting().do(archive.get("F"), only_non_dominated_front=True)
    return archive[nds]


def best_optimum(problem, archive):
    """The single best (fitness-survival) member of an archive (the optimum for single-objective methods).

    Args:
        problem: The problem (needed by pymoo's survival).
        archive: The evaluated population.

    Returns:
        A population of one -- the best solution.
    """
    return FitnessSurvival().do(problem, archive, n_survive=1)
