from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.normalization import ZeroToOneNormalization

from pysamoo.core.defaults import DEFAULT_OBJ_MODELS, DEFAULT_IEQ_CONSTR_MODELS, DEFAULT_EQ_CONSTR_MODELS
from pysamoo.core.surrogate import Surrogate
from pysamoo.core.target import Target


def default_n_doe(n, max=float("inf")):
    return min(2 * n + 1, max)


class SurrogateAssistedAlgorithm(Algorithm):

    def __init__(self,
                 n_initial_doe=None,
                 n_initial_max_doe=100,
                 sampling=LHS(),
                 nth_validate=5,
                 surrogate=None,
                 **kwargs):
        """
        Parameters
        ----------
        n_initial_doe : int
            Number of initial design of experiments. If `None`, the default is 11*n - 1. (but at most `n_max_doe`)

        n_max_doe : int
            If `n_initial_doe` is set to `None`, the maximum number of initial designs.

        sampling : class
            The initial sampling being used for the designs of experiment.

        """
        super().__init__(**kwargs)

        self.n_initial_doe = n_initial_doe
        self.n_initial_max_doe = n_initial_max_doe
        self.initialization = Initialization(sampling)

        # all solutions that have been evaluated so far
        self.archive = Population()

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

            targets = []

            models = DEFAULT_OBJ_MODELS(**defaults)
            for m in range(problem.n_obj):
                target = Target(("F", m), models)
                targets.append(target)

            models = DEFAULT_IEQ_CONSTR_MODELS(**defaults)
            for g in range(problem.n_ieq_constr):
                target = Target(("G", g), models)
                targets.append(target)

            models = DEFAULT_EQ_CONSTR_MODELS(**defaults)
            for h in range(problem.n_eq_constr):
                target = Target(("H", h), models)
                targets.append(target)

            # create the surrogate model
            self.surrogate = Surrogate(problem, targets)

        # set the number of DOE points initially
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))

    def _initialize_infill(self):
        self.infills = self.initialization.do(self.problem, self.n_initial_doe, algorithm=self)
        return self.infills

    def _initialize_advance(self, infills=None, **kwargs):
        self.infills = infills
        self.archive = Population.merge(self.archive, infills)

    def _advance(self, infills=None, **kwargs):
        self.infills = infills
        self.archive = Population.merge(self.archive, infills)


class MyNormalization(ZeroToOneNormalization):

    def forward(self, X):
        return super().forward(X) * 200 - 100

    def backward(self, X):
        return super().backward((X + 100) / 200)
