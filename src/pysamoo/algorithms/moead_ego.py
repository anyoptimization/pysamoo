"""MOEA/D-EGO -- decomposition-based efficient global optimization with a batch infill per iteration."""

from copy import deepcopy

import numpy as np
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from pysurrogate.dace import Exponential
from pysurrogate.models import Kriging

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm, default_n_doe


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

    def __init__(self, ref_dirs=None, n_infills=5, kappa=2.0, pool=200, surrogate=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.ref_dirs = ref_dirs
        self.n_infills = n_infills
        self.kappa = kappa
        self.pool = pool
        self.surrogate_proto = surrogate if surrogate is not None else Kriging(corr=Exponential())

    def _setup(self, problem, **kwargs):
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))
        if self.ref_dirs is None:
            n_partitions = {2: 99, 3: 12}.get(problem.n_obj, 6)
            self.ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=n_partitions)

    def _infill(self):
        X, F = self._archive.get("X", "F")
        problem = self.problem
        xl, xu = problem.xl, problem.xu
        rng = self.random_state

        # one Kriging per objective (fit once, reused across all subproblems)
        models = [deepcopy(self.surrogate_proto) for _ in range(problem.n_obj)]
        for m, model in enumerate(models):
            model.fit(X, F[:, m])

        # candidate pool: LHS + local perturbations of the current non-dominated designs
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        cand = LHS().do(problem, self.pool, random_state=rng).get("X")
        idx = rng.integers(len(nds), size=self.pool)
        local = np.clip(X[nds][idx] + 0.05 * (xu - xl) * rng.standard_normal((self.pool, problem.n_var)), xl, xu)
        cand = np.vstack([cand, local])

        # optimistic per-objective prediction (lower-confidence bound)
        mu = np.column_stack([model.predict(cand).y[:, 0] for model in models])
        sigma = np.column_stack([model.predict(cand, var=True).sigma[:, 0] for model in models])
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
        nds = NonDominatedSorting().do(self._archive.get("F"), only_non_dominated_front=True)
        self.opt = self._archive[nds]
