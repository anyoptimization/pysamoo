"""Base class and helpers for surrogate-assisted algorithms."""

from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.normalization import ZeroToOneNormalization

from pysamoo.core.defaults import DEFAULT_EQ_CONSTR_MODELS, DEFAULT_IEQ_CONSTR_MODELS, DEFAULT_OBJ_MODELS
from pysamoo.core.selection import resolve as resolve_selection
from pysamoo.core.surrogate import Surrogate


def default_n_doe(n_var, cap=float("inf")):
    """Default initial design-of-experiments size for ``n_var`` variables, optionally capped.

    Args:
        n_var: Number of decision variables.
        cap: Upper bound on the returned size (defaults to no cap).

    Returns:
        ``min(2 * n_var + 1, cap)``.
    """
    return min(2 * n_var + 1, cap)


class SurrogateAssistedAlgorithm(Algorithm):
    # whether _setup builds the default multi-target model pool; the EGO-style algorithms that
    # manage their own per-objective models set this to False (and skip the expensive build).
    build_default_surrogate = True

    def __init__(
        self,
        n_initial_doe=None,
        n_initial_max_doe=100,
        sampling=None,
        nth_validate=5,
        surrogate=None,
        selection="full",
        **kwargs,
    ):
        """Base surrogate-assisted algorithm.

        Args:
            n_initial_doe: Number of initial design-of-experiments points. If ``None``, defaults to
                ``2 * n_var + 1`` (capped at ``n_initial_max_doe``).
            n_initial_max_doe: Upper bound on the initial DOE size when ``n_initial_doe`` is ``None``.
            sampling: The sampling operator used to generate the initial designs.
            nth_validate: Re-run full model selection only every nth call to :meth:`revalidate`
                (the model is still refit every iteration in between).
            surrogate: A pre-built :class:`~pysamoo.core.surrogate.Surrogate`. If ``None``, a default
                model pool is constructed and ``selection`` chooses among it.
            selection: Pluggable model-selection strategy for the default surrogate — a name registered
                in :data:`pysamoo.core.selection.STRATEGIES` or a Target factory
                ``(label, models) -> Target``. ``"full"`` cross-validates the whole pool every
                iteration. Ignored if ``surrogate`` is given.
        """
        super().__init__(**kwargs)

        self.selection = selection
        self.n_initial_doe = n_initial_doe
        self.n_initial_max_doe = n_initial_max_doe
        self.initialization = Initialization(sampling if sampling is not None else LHS())

        # all solutions that have been evaluated so far
        self._archive = Population()

        # here always the most recent infill solutions are stored
        self.infills = None

        # the model/surrogate to be used during optimization
        self.surrogate = surrogate

        # each nth iteration when all surrogate models should be revalidated
        self.nth_validate = nth_validate

        # counts calls to revalidate() so re-selection can run only every nth_validate-th time
        self._revalidate_count = 0

    def _setup(self, problem, **kwargs):
        # set the number of DOE points initially
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))

        # build the default multi-target surrogate unless the algorithm manages its own models
        if self.build_default_surrogate and self.surrogate is None:
            self.surrogate = self._build_default_surrogate(problem)

    def _build_default_surrogate(self, problem):
        """Construct the default multi-target surrogate (one model pool per objective/constraint).

        Args:
            problem: The problem whose objective/constraint counts and bounds shape the surrogate.

        Returns:
            A :class:`~pysamoo.core.surrogate.Surrogate` with a selection-driven target per output.
        """
        # the design space boundaries for the problem - used for normalization in the surrogate
        xl, xu = problem.bounds()
        defaults = dict(norm_X=MyNormalization(xl, xu))

        # the model-selection strategy is pluggable: resolve it to a target
        # factory (label, models) -> Target. "full" or any factory.
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

        return Surrogate(problem, targets)

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
        self._revalidate_count += 1
        # validate on the first call and then every nth_validate-th call (so a
        # caller whose first selection happens here — e.g. BO — is covered).
        if not self.nth_validate or (self._revalidate_count - 1) % self.nth_validate == 0:
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
    """Map the design space to ``[-100, 100]`` (symmetric, wider range than [0, 1]).

    Surrogate kernels (RBF/Kriging) are better conditioned on this symmetric, wider range than on the
    plain unit cube, so the default model pool normalizes design inputs through this before fitting.
    """

    def forward(self, X):
        return super().forward(X) * 200 - 100

    def backward(self, X):
        return super().backward((X + 100) / 200)
