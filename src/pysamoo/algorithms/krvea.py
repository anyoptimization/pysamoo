"""K-RVEA -- Kriging-assisted reference-vector guided EA for expensive many-objective optimization."""

from copy import deepcopy

import numpy as np
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from pysurrogate.dace import Exponential
from pysurrogate.models import Kriging

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm, default_n_doe


class _KrigingProblem(Problem):
    """A cheap pymoo problem whose objectives are the per-objective Kriging *mean* predictions."""

    def __init__(self, models, xl, xu):
        super().__init__(n_var=len(xl), n_obj=len(models), xl=xl, xu=xu)
        self.models = models

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.column_stack([m.predict(X).y[:, 0] for m in self.models])


class KRVEA(SurrogateAssistedAlgorithm):
    """K-RVEA (Chugh et al., 2018): RVEA driven by Kriging models, with an adaptive infill criterion.

    Each iteration fits one Kriging model per objective on the archive, runs **RVEA** on those
    (cheap) surrogate predictions for ``w_max`` generations, then selects ``n_infills`` candidates to
    evaluate on the true function. The selection alternates between two K-RVEA criteria, switched by
    how much the set of *active reference vectors* changed since the last iteration:

    * **diversity** (many active vectors changed -> the front is still moving): pick the
      best-aligned candidate for distinct reference vectors, spreading the search;
    * **convergence/uncertainty** (stable): pick the candidates with the highest Kriging
      uncertainty, improving the models where they are least sure.

    Reuses pymoo's RVEA + Das-Dennis reference vectors and the pysurrogate Kriging surrogate
    (Kriging is required here because the uncertainty criterion needs a predictive ``sigma``).

    Args:
        ref_dirs: Reference vectors; ``None`` builds a Das-Dennis set sized to ``n_obj``.
        n_infills: True-function evaluations selected per iteration (``u`` in the paper).
        w_max: RVEA generations run on the surrogate between infills.
        delta: Fraction-of-reference-vectors change above which the diversity criterion is used.
        surrogate: pysurrogate Kriging prototype per objective (default ``Kriging(Exponential())``).
    """

    def __init__(self, ref_dirs=None, n_infills=5, w_max=20, delta=0.05, surrogate=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.ref_dirs = ref_dirs
        self.n_infills = n_infills
        self.w_max = w_max
        self.delta = delta
        self.surrogate_proto = surrogate if surrogate is not None else Kriging(corr=Exponential())
        self._active_prev = None

    def _setup(self, problem, **kwargs):
        # K-RVEA manages its own per-objective Kriging models, so -- like the other EGO-style
        # algorithms here -- it skips the base class's single shared surrogate build.
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))
        if self.ref_dirs is None:
            n_partitions = {2: 99, 3: 12}.get(problem.n_obj, 6)
            self.ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=n_partitions)

    def _infill(self):
        X, F = self._archive.get("X", "F")
        problem = self.problem

        # 1) one Kriging per objective
        models = [deepcopy(self.surrogate_proto) for _ in range(problem.n_obj)]
        for m, model in enumerate(models):
            model.fit(X, F[:, m])

        # 2) optimize the surrogate with RVEA for w_max generations (threaded seed -> reproducible)
        surr = _KrigingProblem(models, problem.xl, problem.xu)
        seed = int(self.random_state.integers(1, 2**31 - 1))
        res = pymoo_minimize(surr, RVEA(ref_dirs=self.ref_dirs), ("n_gen", self.w_max), seed=seed, verbose=False)
        Xc, Fc = res.pop.get("X"), res.pop.get("F")

        # 3) predictive uncertainty (summed sigma) for each candidate
        sigma = np.column_stack([model.predict(Xc, var=True).sigma[:, 0] for model in models]).sum(axis=1)

        # 4) associate candidates to reference vectors by acute angle (translated to the ideal point)
        U = Fc - Fc.min(axis=0)
        Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        cos = Un @ self.ref_dirs.T
        assign = cos.argmax(axis=1)
        active = set(np.unique(assign).tolist())

        # 5) K-RVEA adaptive criterion: how much the active-reference-vector set changed decides
        #    whether to emphasize diversity (spread) or convergence (reduce uncertainty).
        changed = 1.0 if self._active_prev is None else len(active ^ self._active_prev) / len(self.ref_dirs)
        self._active_prev = active

        u = min(self.n_infills, len(Xc))
        if changed > self.delta:
            # diversity: best-aligned candidate per distinct active reference vector
            picks = []
            for rv in sorted(active):
                members = np.where(assign == rv)[0]
                picks.append(int(members[cos[members, rv].argmax()]))
                if len(picks) >= u:
                    break
            sel = np.array(picks[:u], dtype=int)
        else:
            # convergence: the u candidates the models are least certain about
            sel = np.argsort(-sigma)[:u]

        return Population.new(X=Xc[sel])

    def _set_optimum(self):
        nds = NonDominatedSorting().do(self._archive.get("F"), only_non_dominated_front=True)
        self.opt = self._archive[nds]
