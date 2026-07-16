"""EHVI -- Expected Hypervolume Improvement Bayesian optimization for multi-objective problems."""

import numpy as np
from pymoo.core.population import Population
from pymoo.util.display.multi import MultiObjectiveOutput

from pysamoo.algorithms._ego import (
    default_kriging,
    fit_per_objective,
    front_and_hv,
    lhs_local_pool,
    pareto_optimum,
    predict_mu_sigma,
)
from pysamoo.core.algorithm import SurrogateAssistedAlgorithm


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

    # manages its own per-objective Kriging models -> skip the base default-surrogate build.
    build_default_surrogate = False

    def __init__(self, pool=200, n_screen=20, n_samples=32, surrogate=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.pool = pool
        self.n_screen = n_screen
        self.n_samples = n_samples
        self.surrogate_proto = surrogate if surrogate is not None else default_kriging()

    def _infill(self):
        X, F = self._archive.get("X", "F")
        problem = self.problem

        # one Kriging per objective
        models = fit_per_objective(self.surrogate_proto, X, F)

        # current non-dominated front and a reference point (nadir + 10% margin) for the hypervolume
        nds, front, hv = front_and_hv(F)
        hv0 = float(hv(front))

        def hvi(point):
            return max(0.0, float(hv(np.vstack([front, point]))) - hv0)

        # candidate pool: a space-filling LHS *plus* local perturbations of the current
        # non-dominated designs. In higher dimensions a pure LHS pool is too sparse to locate the
        # acquisition optimum, so seeding the neighbourhood of the front is what makes EHVI competitive.
        cand = lhs_local_pool(problem, X[nds], self.pool, self.random_state)
        mu, sigma = predict_mu_sigma(models, cand)

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
        self.opt = pareto_optimum(self._archive)
