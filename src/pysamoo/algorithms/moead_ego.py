"""MOEA/D-EGO -- decomposition-based efficient global optimization with a batch infill per iteration."""

import numpy as np
from pymoo.core.population import Population
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from pysamoo.algorithms._ego import default_kriging, fit_per_objective, lhs_local_pool, pareto_optimum, predict_mu_sigma
from pysamoo.core.algorithm import SurrogateAssistedAlgorithm


class MOEADEGO(SurrogateAssistedAlgorithm):
    """MOEA/D-EGO (Zhang et al., 2010): per-objective Kriging + a decomposition-based batch infill.

    Unlike ParEGO (which refits one scalar surrogate for a *single* random weight each iteration),
    MOEA/D-EGO fits one Kriging model per objective *once* per iteration and reuses them across many
    weight vectors, selecting a whole batch of ``n_infills`` points -- one per (spread) weight vector.
    Each subproblem scores candidates by the Tchebycheff aggregation of an *optimistic* prediction
    ``mu - kappa*sigma`` (a lower-confidence-bound acquisition on the decomposed subproblem), and the
    best not-yet-chosen candidate is taken, giving a diverse batch. Reuses pymoo's Das-Dennis
    reference vectors and the pysurrogate Kriging surrogate.

    Args:
        ref_dirs: Weight vectors; ``None`` builds a Das-Dennis set sized to ``n_obj``.
        n_infills: Points evaluated per iteration (one per selected weight vector).
        kappa: Exploration weight of the optimistic ``mu - kappa*sigma`` prediction.
        pool: LHS candidate-pool size (augmented with local perturbations of the current front).
        surrogate: pysurrogate Kriging prototype per objective (default ``Kriging(Exponential())``).
    """

    # manages its own per-objective Kriging models -> skip the base default-surrogate build.
    build_default_surrogate = False

    def __init__(self, ref_dirs=None, n_infills=5, kappa=2.0, pool=200, surrogate=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.ref_dirs = ref_dirs
        self.n_infills = n_infills
        self.kappa = kappa
        self.pool = pool
        self.surrogate_proto = surrogate if surrogate is not None else default_kriging()

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        if self.ref_dirs is None:
            n_partitions = {2: 99, 3: 12}.get(problem.n_obj, 6)
            self.ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=n_partitions)

    def _infill(self):
        X, F = self._archive.get("X", "F")
        problem = self.problem
        rng = self.random_state

        # one Kriging per objective (fit once, reused across all subproblems)
        models = fit_per_objective(self.surrogate_proto, X, F)

        # candidate pool: LHS + local perturbations of the current non-dominated designs
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        cand = lhs_local_pool(problem, X[nds], self.pool, rng)

        # optimistic per-objective prediction (lower-confidence bound)
        mu, sigma = predict_mu_sigma(models, cand)
        lcb = mu - self.kappa * sigma
        z = lcb.min(axis=0)  # ideal-point estimate on the optimistic prediction

        # one point per (spread) weight vector by minimum Tchebycheff aggregation -> a diverse batch
        picks = np.linspace(0, len(self.ref_dirs) - 1, self.n_infills).round().astype(int)
        chosen: list = []
        for w in self.ref_dirs[picks]:
            g = (w * (lcb - z)).max(axis=1)
            for i in np.argsort(g):
                if int(i) not in chosen:
                    chosen.append(int(i))
                    break
        return Population.new(X=cand[chosen])

    def _set_optimum(self):
        self.opt = pareto_optimum(self._archive)
