"""ParEGO -- Pareto-efficient global optimization via random augmented-Tchebycheff scalarization."""

from copy import deepcopy

import numpy as np
from pymoo.core.population import Population
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.ref_dirs import get_reference_directions

from pysamoo.algorithms._ego import default_kriging, pareto_optimum
from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
from pysamoo.experimental.acquisition import LogEI
from pysamoo.experimental.infill import GlobalEI
from pysamoo.experimental.optimizer import VectorizedGradientDescent


class ParEGO(SurrogateAssistedAlgorithm):
    """ParEGO (Knowles, 2006): single-objective EGO on a per-iteration random scalarization.

    Each infill draws one weight vector ``lambda`` uniformly from a fixed Das-Dennis set, collapses
    the (archive-normalized) objectives to a single value with the *augmented Tchebycheff* function
    ``g = max_i(lambda_i * f_i) + rho * sum_i(lambda_i * f_i)``, fits one Kriging model to those
    scalar values, and maximizes Expected Improvement over the box to pick the next point. Rotating
    the weight each iteration spreads the search across the whole Pareto front while only ever
    optimizing a *single-objective* surrogate -- so it reuses the repo's existing EGO machinery
    wholesale (the pysurrogate Kriging surrogate and the ``experimental`` EI acquisition + optimizer).

    This is the smallest end-to-end multi-objective EGO on the current stack; EHVI/qNEHVI slot into
    the same acquisition seam later.

    Args:
        rho: Weight of the augmentation term in the Tchebycheff scalarization (Knowles uses 0.05).
        surrogate: A pysurrogate ``Model`` for the scalarized value (default ``Kriging(Exponential())``,
            the same surrogate the single-objective BO defaults to). Re-fit fresh every infill.
        infill: The acquisition seam (default ``GlobalEI(VectorizedGradientDescent())``) -- any
            ``Infill`` that maximizes ``acq_func`` over the box.
        acq_func: The acquisition function on the scalar surrogate (default ``LogEI``).
    """

    # ParEGO manages its own single-objective (scalar) surrogate -> skip the base build.
    build_default_surrogate = False

    def __init__(self, rho=0.05, surrogate=None, infill=None, acq_func=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.rho = rho
        self.surrogate_proto = surrogate if surrogate is not None else default_kriging()
        self.infill_strategy = infill if infill is not None else GlobalEI(VectorizedGradientDescent())
        self.acq_func = acq_func if acq_func is not None else LogEI()
        self.weights = None
        self._model = None

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        # a fixed Das-Dennis weight set; one is drawn at random per infill. The partition count is
        # picked so the set is neither tiny nor huge for the common 2-/3-objective cases.
        n_partitions = {2: 100, 3: 15}.get(problem.n_obj, 8)
        self.weights = get_reference_directions("das-dennis", problem.n_obj, n_partitions=n_partitions)

    def _infill(self):
        X, F = self._archive.get("X", "F")

        # normalize the objectives to [0, 1] with the current archive (so the ideal point is 0 and
        # every objective contributes on the same scale to the scalarization)
        z_min, z_max = F.min(axis=0), F.max(axis=0)
        Fn = (F - z_min) / np.maximum(z_max - z_min, 1e-12)

        # draw one weight and collapse to a scalar via augmented Tchebycheff
        lam = self.weights[self.random_state.integers(len(self.weights))]
        d = lam * Fn
        y = d.max(axis=1) + self.rho * d.sum(axis=1)

        # fit a fresh surrogate on the scalarized values (the scalarization changes every infill)
        model = deepcopy(self.surrogate_proto)
        model.fit(X, y)
        self._model = model

        # maximize (Log)EI on the scalar surrogate to choose the next point
        x_best, _ = self.infill_strategy.do(self.problem, lambda: model, X, y, self.acq_func, self.random_state)
        return Population.new(X=x_best[None, :])

    def _set_optimum(self):
        self.opt = pareto_optimum(self._archive)
