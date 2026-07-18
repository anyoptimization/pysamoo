"""TSEMO -- Thompson-Sampling Efficient Multi-objective Optimization."""

import numpy as np
from pymoo.core.population import Population
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pysamoo.algorithms._ego import (
    default_kriging,
    fit_per_objective,
    front_and_hv,
    lhs_local_pool,
    pareto_optimum,
    predict_mu_sigma,
)
from pysamoo.core.algorithm import SurrogateAssistedAlgorithm


class TSEMO(SurrogateAssistedAlgorithm):
    """TSEMO (Bradford et al., 2018): Thompson-sampling multi-objective Bayesian optimization.

    Fits one Kriging model per objective, draws a single Thompson sample from each posterior over a
    candidate pool, and takes the sample's Pareto-optimal candidates -- the search commits to *one*
    plausible objective landscape rather than an average, which naturally balances exploration and
    exploitation. From those sampled-Pareto candidates it greedily selects the ``n_infills`` points
    that most increase the *true* front's hypervolume. Reuses pymoo's HV indicator + non-dominated
    sorting and the pysurrogate Kriging surrogate (the ``sigma`` drives the Thompson draw).

    Args:
        n_infills: True-function evaluations selected per iteration.
        pool: LHS candidate-pool size (augmented with local perturbations of the current front).
        surrogate: pysurrogate Kriging prototype per objective (default ``Kriging(Exponential())``).
    """

    # manages its own per-objective Kriging models -> skip the base default-surrogate build.
    build_default_surrogate = False

    def __init__(self, n_infills=5, pool=200, surrogate=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.n_infills = n_infills
        self.pool = pool
        self.surrogate_proto = surrogate if surrogate is not None else default_kriging()

    def _infill(self):
        X, F = self._archive.get("X", "F")
        problem = self.problem
        rng = self.random_state

        # one Kriging per objective
        models = fit_per_objective(self.surrogate_proto, X, F)

        # current front, reference point and hypervolume
        nds, front, hv = front_and_hv(F)

        # candidate pool: LHS + local perturbations of the current non-dominated designs
        cand = lhs_local_pool(problem, X[nds], self.pool, rng)

        # Thompson sample: one posterior draw per objective at the candidates
        mu, sigma = predict_mu_sigma(models, cand)
        sample = self.thompson_sample(mu, sigma, rng)

        # candidates that are Pareto-optimal under the sampled landscape
        s_nd = NonDominatedSorting().do(sample, only_non_dominated_front=True)

        # greedily pick n_infills of them by hypervolume improvement (on the sample) over the true front
        chosen = self.greedy_hvi_select(sample, front, hv, self.n_infills, list(s_nd))

        # fall back to the best sampled candidates if none improved the hypervolume
        if not chosen:
            chosen = list(s_nd[: self.n_infills])
        return Population.new(X=cand[chosen])

    @staticmethod
    def thompson_sample(mu, sigma, random_state):
        """One Thompson posterior draw per objective: ``mu + sigma * z`` with ``z ~ N(0, I)``.

        Committing to a single sampled landscape (rather than the mean) is what makes TSEMO balance
        exploration and exploitation. Exposed as a static method for the fidelity tests.

        Args:
            mu: Predicted objective means, shape ``(n, n_obj)``.
            sigma: Predictive standard deviations, shape ``(n, n_obj)``.
            random_state: A numpy ``Generator`` for the draw.

        Returns:
            The sampled objective landscape, shape ``(n, n_obj)``.
        """
        return mu + sigma * random_state.standard_normal(mu.shape)

    @staticmethod
    def greedy_hvi_select(sample, front, hv, n_infills, candidates):
        """Greedily select candidate indices that most increase the sampled front's hypervolume.

        Each step adds the candidate with the largest marginal hypervolume improvement over the set
        chosen so far (plus the true front) -- so the batch is diverse in objective space. Exposed
        for the fidelity tests.

        Args:
            sample: The sampled objective values, shape ``(n, n_obj)``.
            front: The current true non-dominated front, shape ``(k, n_obj)``.
            hv: A ``pymoo`` ``HV`` indicator with the reference point already set.
            n_infills: Maximum number of candidates to select.
            candidates: The candidate indices to select from.

        Returns:
            A list of the selected candidate indices (at most ``n_infills``).
        """
        pool_idx, chosen = list(candidates), []
        cur = front
        for _ in range(min(n_infills, len(pool_idx))):
            best_i, best_gain = None, -np.inf
            hv_cur = float(hv(cur))
            for i in pool_idx:
                gain = float(hv(np.vstack([cur, sample[i]]))) - hv_cur
                if gain > best_gain:
                    best_gain, best_i = gain, i
            chosen.append(best_i)
            pool_idx.remove(best_i)
            cur = np.vstack([cur, sample[best_i]])
        return chosen

    def _set_optimum(self):
        self.opt = pareto_optimum(self._archive)
