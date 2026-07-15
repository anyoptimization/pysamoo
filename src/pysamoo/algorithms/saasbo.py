"""SAASBO -- Sparse Axis-Aligned Subspace Bayesian Optimization (a MAP / shrinkage approximation)."""

from copy import deepcopy

from pymoo.core.population import Population
from pymoo.util.display.single import SingleObjectiveOutput
from pysurrogate.dace import Exponential
from pysurrogate.models import Kriging

from pysamoo.algorithms._ego import best_optimum
from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
from pysamoo.experimental.acquisition import LogEI
from pysamoo.experimental.infill import GlobalEI
from pysamoo.experimental.optimizer import GeneticAlgorithm


class SAASBO(SurrogateAssistedAlgorithm):
    """SAASBO (Eriksson & Jankowiak, 2021): Bayesian optimization with a *sparse axis-aligned* GP.

    High-dimensional BO fails because a full-ARD GP over-fits its per-dimension length-scales with few
    samples. SAASBO puts a strong shrinkage prior on those length-scales so most dimensions stay
    "off" (long length-scale) unless the data demands otherwise, concentrating the model on the few
    active axes. The reference method samples the prior with NUTS; here we use the cheaper MAP form
    the surrogate already supports -- an ARD Kriging model with a shrinkage ``theta_prior`` on the
    (log) length-scales -- and drive it with standard Expected Improvement. Reuses the pysurrogate
    ARD Kriging + shrinkage prior and the experimental EI acquisition.

    .. note::
        This is a **scaffold, not yet competitive**. Faithful SAASBO relies on a *sparse GP with NUTS
        hyperparameter sampling*; the MAP ``theta_prior`` here is a coarse substitute and does not
        reliably beat a plain baseline on high-dimensional problems. Making SAASBO competitive needs
        the sparse-GP / MCMC infrastructure that ``pysurrogate`` does not yet have -- so this class is
        shipped as the algorithm structure (runnable and reproducible) with that modeling work called
        out, and it carries no performance assertion until the surrogate side lands.

    Args:
        theta_prior: ``(mean, std)`` shrinkage prior on the log length-scale hyperparameters; a small
            mean/std pulls dimensions toward "inactive" (the sparsity that makes ARD work in high-d).
        surrogate: pysurrogate surrogate (default: ARD ``Kriging(Exponential())`` with ``theta_prior``).
        infill: acquisition seam (default ``GlobalEI(GeneticAlgorithm())`` -- derivative-free, robust).
        acq_func: acquisition function (default ``LogEI``).
    """

    # manages its own sparse-ARD Kriging model -> skip the base default-surrogate build.
    build_default_surrogate = False

    def __init__(self, theta_prior=(0.0, 0.01), surrogate=None, infill=None, acq_func=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else SingleObjectiveOutput(), **kwargs)
        self.theta_prior = theta_prior
        if surrogate is not None:
            self.surrogate_proto = surrogate
        else:
            self.surrogate_proto = Kriging(corr=Exponential(), ARD=True, theta_prior=theta_prior)
        self.infill_strategy = infill if infill is not None else GlobalEI(GeneticAlgorithm())
        self.acq_func = acq_func if acq_func is not None else LogEI()
        self._model = None

    def _infill(self):
        X, F = self._archive.get("X", "F")
        y = F[:, 0]

        # fit the sparse-ARD surrogate on all data
        model = deepcopy(self.surrogate_proto)
        model.fit(X, y)
        self._model = model

        # maximize Expected Improvement over the box
        x_best, _ = self.infill_strategy.do(self.problem, lambda: model, X, y, self.acq_func, self.random_state)
        return Population.new(X=x_best[None, :])

    def _set_optimum(self):
        self.opt = best_optimum(self.problem, self._archive)
