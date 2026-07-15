"""TuRBO -- Trust-Region Bayesian Optimization for higher-dimensional single-objective problems."""

from copy import deepcopy

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.population import Population
from pymoo.util.display.single import SingleObjectiveOutput
from pysurrogate.dace import Exponential
from pysurrogate.models import Kriging

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
from pysamoo.experimental.acquisition import LogEI


class TuRBO(SurrogateAssistedAlgorithm):
    """TuRBO-1 (Eriksson et al., 2019): Bayesian optimization inside an adaptive trust region.

    Standard global BO over-explores in higher dimensions; TuRBO restricts the surrogate and the
    acquisition to a hyper-rectangular *trust region* centered on the incumbent. Each infill fits a
    Kriging model, generates candidates inside the trust region (perturbing only a random subset of
    coordinates per candidate, as in TuRBO), and picks the one with the best (Log)Expected
    Improvement. The trust-region side length ``L`` adapts to progress: it doubles after ``succ_tol``
    consecutive improvements and halves after ``fail_tol`` failures; when it collapses below
    ``length_min`` the region restarts at ``length_init``. Reuses the pysurrogate Kriging surrogate
    and the experimental ``LogEI`` acquisition.

    Args:
        n_candidates: Trust-region candidates scored per infill (default ``min(1000, 100*d)``).
        length_init: Initial trust-region side as a fraction of the box width.
        length_min: Side below which the trust region restarts.
        length_max: Maximum trust-region side.
        succ_tol: Consecutive improvements that grow the region.
        fail_tol: Consecutive failures that shrink it (default ``max(4, d)``).
        surrogate: pysurrogate Kriging prototype (default ``Kriging(Exponential())``).
        acq_func: Acquisition scored over the trust-region candidates (default ``LogEI``).
    """

    # manages its own single-objective Kriging -> skip the base default-surrogate build.
    build_default_surrogate = False

    def __init__(
        self,
        n_candidates=None,
        length_init=0.8,
        length_min=0.5**7,
        length_max=1.6,
        succ_tol=3,
        fail_tol=None,
        surrogate=None,
        acq_func=None,
        output=None,
        **kwargs,
    ):
        super().__init__(output=output if output is not None else SingleObjectiveOutput(), **kwargs)
        self.n_candidates = n_candidates
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.succ_tol = succ_tol
        self.fail_tol = fail_tol
        self.surrogate_proto = surrogate if surrogate is not None else Kriging(corr=Exponential())
        self.acq_func = acq_func if acq_func is not None else LogEI()
        self.L = length_init
        self.success = 0
        self.failure = 0
        self._restart = False
        self._model = None

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        if self.n_candidates is None:
            self.n_candidates = min(1000, 100 * problem.n_var)
        if self.fail_tol is None:
            self.fail_tol = max(4, problem.n_var)

    def _infill(self):
        X, F = self._archive.get("X", "F")
        y = F[:, 0]
        problem = self.problem
        xl, xu, d = problem.xl, problem.xu, problem.n_var
        span = xu - xl
        rng = self.random_state

        # restart: the trust region collapsed, so re-seed the search from a fresh random region
        # instead of shrinking forever around a stuck incumbent.
        if self._restart:
            self._restart = False
            self.L = self.length_init
            return Population.new(X=(xl + rng.random(d) * span)[None, :])

        # fit the surrogate on all data
        model = deepcopy(self.surrogate_proto)
        model.fit(X, y)
        self._model = model

        # trust region (side L * box width) centered on the incumbent
        x_center = X[int(y.argmin())]
        half = 0.5 * self.L * span
        tr_lb = np.maximum(x_center - half, xl)
        tr_ub = np.minimum(x_center + half, xu)

        # candidates inside the trust region: each perturbs only a random subset of coordinates
        n = self.n_candidates
        pert = tr_lb + (tr_ub - tr_lb) * rng.random((n, d))
        prob = min(20.0 / d, 1.0)
        mask = rng.random((n, d)) < prob
        rows = np.where(~mask.any(axis=1))[0]  # every candidate must perturb at least one coordinate
        if len(rows):
            mask[rows, rng.integers(0, d, size=len(rows))] = True
        cand = np.repeat(x_center[None, :], n, axis=0).copy()
        cand[mask] = pert[mask]

        # score by (Log)EI over the candidate set; the acquisition is in minimization form
        pred = model.predict(cand, var=True)
        acq = self.acq_func.calc(pred.y[:, 0], pred.sigma[:, 0], f_min=float(y.min()))
        x_best = cand[int(np.argmin(acq))]
        return Population.new(X=x_best[None, :])

    def _advance(self, infills=None, **kwargs):
        prev = float(self._archive.get("F")[:, 0].min()) if len(self._archive) else float("inf")
        super()._advance(infills, **kwargs)
        cur = float(self._archive.get("F")[:, 0].min())

        if cur < prev - 1e-12 * max(1.0, abs(prev)):
            self.success, self.failure = self.success + 1, 0
        else:
            self.success, self.failure = 0, self.failure + 1

        if self.success >= self.succ_tol:
            self.L, self.success = min(2.0 * self.L, self.length_max), 0
        if self.failure >= self.fail_tol:
            self.L, self.failure = self.L / 2.0, 0
        if self.L < self.length_min:  # trust region collapsed -> restart from a fresh region next infill
            self._restart, self.success, self.failure = True, 0, 0

    def _set_optimum(self):
        self.opt = FitnessSurvival().do(self.problem, self._archive, n_survive=1)
