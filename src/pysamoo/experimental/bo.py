"""Bayesian optimization over a pluggable pysurrogate surrogate (model selection by default)."""

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.column import Column
from pymoo.util.display.single import SingleObjectiveOutput
from pysurrogate.dace import Exponential
from pysurrogate.models import Kriging

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm, default_n_doe
from pysamoo.experimental.acquisition import AcquisitionProblem, LogEI
from pysamoo.experimental.infill import GlobalEI, Hybrid
from pysamoo.experimental.optimizer import VectorizedGradientDescent

# ---------------------------------------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------------------------------------


class EGOOutput(SingleObjectiveOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = SingleObjectiveOutput()

        self.f_new = Column(name="f_new")
        self.acq = Column(name="acq")

    def initialize(self, algorithm):
        self.output.initialize(algorithm)
        self.columns = self.output.columns + [self.f_new, self.acq]

    def update(self, algorithm):
        bo = algorithm
        self.output.update(bo)

        if algorithm.acq is not None:
            self.f_new.set(bo.infills.get("F").min())
            self.acq.set(bo.infills.get("acq").min())


# ---------------------------------------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------------------------------------


def default_surrogate():
    """The default BO surrogate: a single Exponential-kernel Kriging model.

    The method-search benchmark (``experimental/benchmark.py``) showed a *fixed* ``Kriging[exp]``
    beats cross-validated ``AutoModel`` over the Kriging fleet on the ECDF score **and runs
    ~2x faster** -- the per-infill model selection is not worth its cost in the low-data BO regime,
    and the Exponential kernel's heavier tails generalize well across the difficulty spectrum
    (Sphere/Rosenbrock/Rastrigin/RotatedEllipsoid). For per-problem kernel selection pass
    ``surrogate=AutoModel(default_kriging())`` explicitly; any pysurrogate Model works.
    """
    return Kriging(corr=Exponential())


class BayesianOptimization(SurrogateAssistedAlgorithm):
    def __init__(
        self,
        acq_func=LogEI(),
        optimizer=VectorizedGradientDescent(),
        infill=None,
        surrogate=None,
        nth_optimize=10,
        output=EGOOutput(),
        **kwargs,
    ):

        super().__init__(output=output, **kwargs)
        self.default_termination = DefaultSingleObjectiveTermination()

        # optimizer: any pysamoo.experimental.optimizer.Optimizer instance, used by the default
        # GlobalEI infill. VectorizedGradientDescent climbs EI by the surrogate's analytic
        # mean/variance gradients. It requires an EI acquisition; for a non-EI one (POI/UCB) pass
        # GeneticAlgorithm.
        self.optimizer = optimizer
        self.acq_func = acq_func
        self.acq = None

        # infill_strategy: the pluggable "where to sample next" strategy
        # (pysamoo.experimental.infill.Infill). Default Hybrid = global EI exploration that hands off
        # to a local quadratic-Newton refinement once EI stalls (and back again), which drives the
        # incumbent far deeper than pure global EI in higher dimensions. Pass GlobalEI(optimizer) for
        # the plain global-EI step, or LocalQuadratic() for pure local refinement. (Named *_strategy
        # to avoid shadowing pymoo Algorithm's own ``infill()`` method.)
        self.infill_strategy = infill if infill is not None else Hybrid(GlobalEI(optimizer))

        # surrogate: ANY pysurrogate Model -- it IS the surrogate, used directly via fit/predict.
        # Default is model selection over the Kriging fleet (default_surrogate()); pass e.g.
        # ``Kriging(corr=Gaussian())`` for a fixed, faster surrogate, or any other pysurrogate Model.
        self.surrogate = surrogate if surrogate is not None else default_surrogate()

        # nth_optimize: the surrogate is fit ONCE on the DOE (which also runs model selection), then
        # *refit* with the new points each infill. Re-optimizing theta every refit is wasteful, so
        # only do it every nth refit (optimize=True) and otherwise refit cheaply (optimize=False).
        # 1 = optimize every refit; None = never re-optimize after the first fit.
        self.nth_optimize = nth_optimize
        self._n_seen = 0
        self._n_refit = 0

        # the fitted surrogate for the current infill (set lazily in _infill by get_model()).
        self._model = None

    def _setup(self, problem, **kwargs):
        # BO uses its own pysurrogate surrogate (fit lazily in _infill); it never uses the base
        # class's surrogate layer, so skip building it and only fix the initial DOE size.
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))

    def _infill(self):

        # all evaluated points so far
        X, F = self._archive.get("X", "F")
        y = F[:, 0]
        problem = self.problem

        # The surrogate is a pysurrogate Model used directly (fit/refit/predict on the object itself
        # -- if it is an AutoModel it runs selection inside its fit). It is fit LAZILY via
        # get_model() so a purely local infill step (LocalQuadratic) skips it. The FIRST call fits;
        # later calls refit only the new points, re-optimizing theta every nth_optimize-th refit.
        def get_model():
            if self._model is None:
                self._model = self.surrogate.fit(X, y)
                self._n_seen = len(X)
            elif len(X) > self._n_seen:
                self._n_refit += 1
                optimize = self.nth_optimize is not None and self._n_refit % self.nth_optimize == 0
                self.surrogate.refit(X[self._n_seen :], y[self._n_seen :], optimize=optimize)
                self._n_seen = len(X)
            return self._model

        # EI improves over the incumbent -- the best objective observed so far.
        f_min = float(y.min())

        # the pluggable infill strategy chooses the next point (GlobalEI fits + maximizes the
        # acquisition; Hybrid/LocalQuadratic add a surrogate-free local quadratic-Newton step).
        x_best, acq_val = self.infill_strategy.do(problem, get_model, X, y, self.acq_func, self.random_state)

        # AcquisitionProblem is kept only for the output/visualization, when a model was fit.
        if self._model is not None:
            self.acq = AcquisitionProblem(problem, self._model, self.acq_func, f_min=f_min)
        return Population.new(X=x_best[None, :], acq=np.array([[acq_val]]))[[0]]

    def _set_optimum(self):
        self.opt = FitnessSurvival().do(self.problem, self._archive, n_survive=1)


class EGOVisualization(Callback):
    def notify(self, algorithm):
        problem = algorithm.problem
        if problem.n_var > 1 or problem.n_obj > 1 or algorithm._model is None:
            return

        fig = plt.figure()

        gs = fig.add_gridspec(4, 1)
        plt_func = fig.add_subplot(gs[:3])
        plt_acq = fig.add_subplot(gs[3])

        X = algorithm.pop.get("X")
        F = problem.evaluate(X)
        infill = algorithm.infills[0]
        plt_func.scatter(X, F, color="red")
        acq = algorithm.acq

        mesh = np.linspace(problem.xl[0], problem.xu[0], 1000)[:, None]

        gp = algorithm._model
        pred = gp.predict(mesh, var=True)
        mu, sigma = pred.y, pred.sigma

        plt_func.fill_between(mesh[:, 0], (mu - 2 * sigma)[:, 0], (mu + 2 * sigma)[:, 0], alpha=0.2, color="k")

        plt_func.scatter(infill.X, infill.F, color="red", s=100, marker="x")
        plt_func.plot(mesh, mu, color="red")

        plt_func.axvline(x=algorithm.infills[0].X, color="black", linestyle="dashed")

        plt_func.plot(mesh, problem.evaluate(mesh), color="black")

        plt_acq.plot(mesh, acq.evaluate(mesh), color="blue")
        plt_acq.scatter(infill.X, acq.evaluate(infill.X), color="red", s=100, marker="x")

        plt.show()
