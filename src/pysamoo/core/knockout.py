"""Additive-noise perturbation of a population's outputs for uncertainty-aware comparisons."""

import numpy as np
from pymoo.core.population import Population

from pysamoo.core.tcv import TotalConstraintViolation


def noisy(sols, error, random_state=None):
    """Return a copy of a population with Gaussian noise added to selected outputs.

    Used by GPSAF to model surrogate prediction uncertainty when comparing solutions.

    Args:
        sols: The population to perturb (its X/F/G/H are copied, not mutated).
        error: Mapping from ``(output_type, column)`` to the noise standard deviation.
        random_state: A numpy ``Generator`` for the noise (falls back to the global RNG).

    Returns:
        A new population with the perturbed outputs and refreshed constraint violation.
    """
    rng = random_state if random_state is not None else np.random
    out = {}
    for type in ["X", "F", "G", "H"]:
        out[type] = np.copy(sols.get(type))

    for (type, k), std in error.items():
        out[type][:, k] += rng.normal(loc=0.0, scale=std, size=len(sols))

    perturbed = Population.new(**out)
    TotalConstraintViolation().do(perturbed, inplace=True)
    return perturbed
