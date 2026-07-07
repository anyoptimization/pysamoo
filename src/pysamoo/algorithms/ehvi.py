"""EHVI -- Expected Hypervolume Improvement Bayesian optimization for multi-objective problems."""

from copy import deepcopy

import numpy as np
from pymoo.core.population import Population
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pysurrogate.dace import Exponential
from pysurrogate.models import Kriging

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm, default_n_doe


class EHVI(SurrogateAssistedAlgorithm):
    """Expected Hypervolume Improvement BO: pick the point that most improves the Pareto hypervolume.

    Fits one Kriging model per objective on the archive, then chooses the next point by maximizing
    the *Expected Hypervolume Improvement* -- the expected growth of the current non-dominated front's
    hypervolume if the point were evaluated. EHVI is estimated by Monte-Carlo (sample the independent
    per-objective Gaussian posteriors, average the hypervolume improvement), which needs no exact-EHVI
    cell decomposition and works for any number of objectives.

    For speed the candidate pool is first *screened* by an optimistic ``mu - sigma`` point's
    hypervolume improvement (cheap, one HV call each), and the Monte-Carlo estimate is computed only
    on the most promising ``n_screen`` candidates. Reuses pymoo's HV indicator and non-dominated
    sorting and the pysurrogate Kriging surrogate (Kriging supplies the predictive ``sigma``).

    Args:
        pool: LHS candidate-pool size drawn per infill.
        n_screen: How many pool candidates (best optimistic HVI) get the Monte-Carlo EHVI estimate.
        n_samples: Monte-Carlo posterior samples per screened candidate.
        surrogate: pysurrogate Kriging prototype per objective (default ``Kriging(Exponential())``).
    """

    def __init__(self, pool=200, n_screen=20, n_samples=32, surrogate=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.pool = pool
        self.n_screen = n_screen
        self.n_samples = n_samples
        self.surrogate_proto = surrogate if surrogate is not None else Kriging(corr=Exponential())

    def _setup(self, problem, **kwargs):
        # manages its own per-objective Kriging models -> skip the base single-surrogate build.
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))

    def _infill(self):
        X, F = self._archive.get("X", "F")
        problem = self.problem
        n_obj = problem.n_obj

        # one Kriging per objective
        models = [deepcopy(self.surrogate_proto) for _ in range(n_obj)]
        for m, model in enumerate(models):
            model.fit(X, F[:, m])

        # current non-dominated front and a reference point (nadir + 10% margin) for the hypervolume
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        front = F[nds]
        z_min, z_max = F.min(axis=0), F.max(axis=0)
        ref = z_max + 0.1 * np.maximum(z_max - z_min, 1e-9)
        hv = HV(ref_point=ref)
        hv0 = float(hv(front))

        def hvi(point):
            return max(0.0, float(hv(np.vstack([front, point]))) - hv0)

        # candidate pool: a space-filling LHS *plus* local perturbations of the current
        # non-dominated designs. In higher dimensions a pure LHS pool is too sparse to locate the
        # acquisition optimum, so seeding the neighbourhood of the front is what makes EHVI competitive.
        xl, xu = problem.xl, problem.xu
        cand = LHS().do(problem, self.pool, random_state=self.random_state).get("X")
        elite_X = X[nds]
        idx = self.random_state.integers(len(elite_X), size=self.pool)
        local = np.clip(
            elite_X[idx] + 0.05 * (xu - xl) * self.random_state.standard_normal((self.pool, problem.n_var)), xl, xu
        )
        cand = np.vstack([cand, local])
        mu = np.column_stack([model.predict(cand).y[:, 0] for model in models])
        sigma = np.column_stack([model.predict(cand, var=True).sigma[:, 0] for model in models])

        # cheap screen: the optimistic point mu - sigma (minimization) rewards both a good mean and
        # high uncertainty, so its HVI is a fast proxy for where EHVI is worth estimating.
        opt_hvi = np.array([hvi(mu[i] - sigma[i]) for i in range(len(cand))])
        top = np.argsort(-opt_hvi)[: self.n_screen]

        # Monte-Carlo EHVI on the screened candidates
        ehvi = np.array(
            [self.expected_hvi(mu[i], sigma[i], front, hv, hv0, self.n_samples, self.random_state) for i in top]
        )

        x_best = cand[top[int(ehvi.argmax())]]
        return Population.new(X=x_best[None, :])

    @staticmethod
    def expected_hvi(mu, sigma, front, hv, hv0, n_samples, random_state):
        """Monte-Carlo Expected Hypervolume Improvement of one Gaussian-predicted candidate.

        Averages ``max(0, HV(front + sample) - hv0)`` over ``n_samples`` draws from the independent
        per-objective Gaussian posterior ``N(mu, diag(sigma^2))``. Exposed as a static method so the
        acquisition can be checked against exact 2-objective EHVI / grid quadrature (fidelity tests).

        Args:
            mu: Predicted objective means of the candidate, shape ``(n_obj,)``.
            sigma: Predictive standard deviations of the candidate, shape ``(n_obj,)``.
            front: The current non-dominated objective vectors, shape ``(k, n_obj)``.
            hv: A ``pymoo`` ``HV`` indicator with the reference point already set.
            hv0: The hypervolume of ``front`` (so it is not recomputed per sample).
            n_samples: Number of Monte-Carlo posterior samples.
            random_state: A numpy ``Generator`` for the samples.

        Returns:
            The Monte-Carlo EHVI estimate (a float).
        """
        samples = mu + sigma * random_state.standard_normal((n_samples, len(mu)))
        return float(np.mean([max(0.0, float(hv(np.vstack([front, s]))) - hv0) for s in samples]))

    def _set_optimum(self):
        nds = NonDominatedSorting().do(self._archive.get("F"), only_non_dominated_front=True)
        self.opt = self._archive[nds]
