"""Base class and helpers for surrogate-assisted algorithms."""

from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.normalization import ZeroToOneNormalization

from pysamoo.core.defaults import DEFAULT_EQ_CONSTR_MODELS, DEFAULT_IEQ_CONSTR_MODELS, DEFAULT_OBJ_MODELS
from pysamoo.core.selection import resolve as resolve_selection
from pysamoo.core.surrogate import Surrogate


def default_n_doe(n, max=float("inf")):
    return min(2 * n + 1, max)


class SurrogateAssistedAlgorithm(Algorithm):
    def __init__(
        self,
        n_initial_doe=None,
        n_initial_max_doe=100,
        sampling=LHS(),
        nth_validate=5,
        surrogate=None,
        selection="full",
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_initial_doe : int
            Number of initial design of experiments. If `None`, the default is 11*n - 1. (but at most `n_max_doe`)

        n_max_doe : int
            If `n_initial_doe` is set to `None`, the maximum number of initial designs.

        sampling : class
            The initial sampling being used for the designs of experiment.

        selection : str or callable
            Pluggable model-selection strategy for the default surrogate — a name
            registered in :data:`pysamoo.core.selection.STRATEGIES` or a Target
            factory ``(label, models) -> Target``. ``"full"`` cross-validates the
            whole pool every iteration (most accurate, slowest); ``"racing"`` keeps
            an adaptive shrinking active set (:class:`~pysamoo.core.racing.RacingTarget`)
            — far faster on large archives at near-identical accuracy. Ignored if
            ``surrogate`` is given.

        """
        super().__init__(**kwargs)

        self.selection = selection
        self.n_initial_doe = n_initial_doe
        self.n_initial_max_doe = n_initial_max_doe
        self.initialization = Initialization(sampling)

        # all solutions that have been evaluated so far
        self._archive = Population()

        # here always the most recent infill solutions are stored
        self.infills = None

        # the model/surrogate to be used during optimization
        self.surrogate = surrogate

        # a solution set which has not been evaluated yet on the models
        self.validation = Population()

        # each nth iteration when all surrogate models should be revalidated
        self.nth_validate = nth_validate

    def _setup(self, problem, **kwargs):

        # initialize the default surrogate for the algorithm
        if self.surrogate is None:
            # the design space boundaries for the problem - used for normalization in the surrogate
            xl, xu = problem.bounds()
            defaults = dict(norm_X=MyNormalization(xl, xu))

            # the model-selection strategy is pluggable: resolve it to a target
            # factory (label, models) -> Target. "full", "racing", or any factory.
            make_target = resolve_selection(self.selection)

            targets = []

            models = DEFAULT_OBJ_MODELS(**defaults)
            for m in range(problem.n_obj):
                targets.append(make_target(("F", m), models))

            models = DEFAULT_IEQ_CONSTR_MODELS(**defaults)
            for g in range(problem.n_ieq_constr):
                targets.append(make_target(("G", g), models))

            models = DEFAULT_EQ_CONSTR_MODELS(**defaults)
            for h in range(problem.n_eq_constr):
                targets.append(make_target(("H", h), models))

            # create the surrogate model
            self.surrogate = Surrogate(problem, targets)

        # set the number of DOE points initially
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))

    def revalidate(self, *args, **kwargs):
        """Re-run model selection lazily.

        Cross-validating the whole candidate pool and re-picking the best model
        (``surrogate.validate``) is the dominant cost, yet the winner rarely
        changes from one infill to the next. ``nth_validate`` decouples how often
        we *re-select* from how often we *refit*: the full selection runs only
        every ``nth_validate``-th call; in between this is a no-op and the current
        best model is reused (algorithms still ``surrogate.fit`` it on the new data
        every iteration). ``nth_validate=1`` re-selects every iteration (original
        behaviour); ``None``/``0`` is treated the same.
        """
        self._revalidate_count = getattr(self, "_revalidate_count", 0) + 1
        if not self.nth_validate or self._revalidate_count % self.nth_validate == 0:
            self.surrogate.validate(*args, **kwargs)

    def _initialize_infill(self):
        # Thread the run's Generator into sampling so the initial DOE is
        # reproducible (pymoo's samplers default to a fresh RNG otherwise).
        self.infills = self.initialization.do(
            self.problem, self.n_initial_doe, algorithm=self, random_state=self.random_state
        )
        return self.infills

    def _initialize_advance(self, infills=None, **kwargs):
        self.infills = infills
        self._archive = Population.merge(self._archive, infills)

    def _advance(self, infills=None, **kwargs):
        self.infills = infills
        self._archive = Population.merge(self._archive, infills)


class MyNormalization(ZeroToOneNormalization):
    def forward(self, X):
        return super().forward(X) * 200 - 100

    def backward(self, X):
        return super().backward((X + 100) / 200)
