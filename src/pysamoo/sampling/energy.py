"""Energy-based constrained sampling."""

import warnings

import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize, normalize
from pymoo.util.ref_dirs.energy import calc_potential_energy_with_grad
from pymoo.util.ref_dirs.optimizer import Adam

from pysamoo.sampling.niching import NichingConstrainedSampling
from pysamoo.sampling.rejection import RejectionConstrainedSampling


class EnergyConstrainedSampling(Sampling):
    def __init__(self, func_eval_constr, n_max_iter=10000):
        super().__init__()
        self.func_eval_constr = func_eval_constr
        self.n_max_iter = n_max_iter

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        constr = self.func_eval_constr
        d = problem.n_var**2

        X = RejectionConstrainedSampling(constr).do(problem, n_samples).get("X")
        if len(X) < n_samples:
            X = NichingConstrainedSampling(constr).do(problem, n_samples).get("X")

        if len(X) == 0:
            raise RuntimeError("No feasible solution could be found!")
        elif len(X) < n_samples:
            warnings.warn("Fewer feasible solutions than requested could be found.", stacklevel=2)

        X = normalize(X, xl, xu)

        optimizer = Adam(alpha=0.005)

        _, grad = calc_potential_energy_with_grad(X, d)

        # a fixed number of energy-minimization steps (the potential energy is monotone under Adam here)
        for _ in range(self.n_max_iter):
            _X = optimizer.next(X, grad)
            _CV = constr(denormalize(_X, xl, xu))
            feasible = np.logical_and(_CV <= 0, np.all(np.logical_and(_X >= 0, _X <= 1), axis=1))

            X[feasible] = _X[feasible]

            _, grad = calc_potential_energy_with_grad(X, d)

        X = denormalize(X, xl, xu)

        return X
