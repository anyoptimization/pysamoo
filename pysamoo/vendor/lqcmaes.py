import warnings

import numpy as np
from cma.fitness_models import LQModel, SurrogatePopulationSettings
from cma.logger import LoggerDummy

from pymoo.algorithms.soo.nonconvex.cmaes import SimpleCMAES
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere


class lqCMAES(SimpleCMAES):

    def __init__(self, **kwargs):
        super().__init__(advance_after_initialization=True, **kwargs)
        self.surrogate = SurrogatePopulation(None)

    def _infill(self):
        pass

    def _advance(self, **kwargs):
        X = super()._infill().get("X")
        X = self.norm.forward(X)

        surr = self.surrogate(X)

        f = None
        infills = []
        while True:
            try:
                x = surr.send(f)
                x = self.norm.backward(x)

                ind = Individual(X=x)
                self.evaluator.eval(self.problem, ind)
                infills.append(ind)
                f = ind.F[0]
            except StopIteration as ex:
                F = ex.value
                break

        self.es.tell(X, F)
        self.pop = Population.create(*infills)

        if self.es.stop():
            self.termination.force_termination = True


class SurrogatePopulation(object):

    def __init__(self,
                 fitness,
                 model=None,
                 model_max_size_factor=None,
                 tau_truth_threshold=None,
                 ):

        self.fitness = fitness
        self.model = model if model else LQModel()
        # set 2 parameters of settings from locals() which are not attributes of self
        self.settings = SurrogatePopulationSettings(locals(), 2, self)
        self.count = 0
        self.evaluations = 0
        self.logger = LoggerDummy(self, labels=['tau0', 'tau1', 'evaluated ratio'])
        self.logger_eigenvalues = LoggerDummy(self.model, ['eigenvalues'])

    class EvaluationManager:

        def __init__(self, X):
            """all is based on the population (list of solutions) `X`"""
            self.X = X
            self.evaluated = len(X) * [False]
            self.fvalues = len(X) * [np.nan]

        def add_eval(self, i, fval):
            """add fitness(self.X[i]), not in use"""
            self.fvalues[i] = fval
            self.evaluated[i] = True

        def eval(self, i, fitness, model_add):
            """add fitness(self.X[i]) to model data, mainly for internal use"""
            if self.evaluated[i]:  # need to decide what to do in this case
                raise ValueError("i=%d, evals=%d, len=%d" % (i, self.evaluations, len(self)))
            self.fvalues[i] = yield self.X[i]

            self.evaluated[i] = True
            model_add(self.X[i], self.fvalues[i])

        def eval_sequence(self, number, fitness, model_add, idx=None):
            """evaluate unevaluated entries of X[idx] until `number` entries are
            evaluated *overall*.

            Assumes that ``sorted(idx) == list(range(len(self.X)))``.

            ``idx`` defines the evaluation sequence.

            The post condition is ``self.evaluations == min(number, len(self.X))``.
            """
            if idx is None:
                idx = range(len(self.X))
            assert len(self.evaluated) == len(self.X)
            if not self.evaluations < number:
                warnings.warn("Expected evaluations=%d < number=%d, popsize=%d"
                              % (self.evaluations, number, len(self.X)))
            self.last_evaluations = number - self.evaluations  # used in surrogate loop
            for i in idx:
                if self.evaluations >= number:
                    break
                if not self.evaluated[i]:
                    yield from self.eval(i, fitness, model_add)
            else:
                if self.evaluations < number and self.evaluations < len(self.X):
                    warnings.warn("After eval: evaluations=%d < number=%d, popsize=%d"
                                  % (self.evaluations, number, len(self.X)))
                return
            assert self.evaluations == number or self.evaluations == len(self.X) < number

        def surrogate_values(self, model_eval, true_values_if_all_available=True,
                             f_offset=None):
            """return surrogate values of `model_eval` with smart offset.
            """
            if true_values_if_all_available and self.evaluations == len(self.X):
                return self.fvalues
            F_model = [model_eval(x) for x in self.X]
            if f_offset is None:
                f_offset = np.nanmin(self.fvalues)  # must be added last to prevent numerical erasion
            if np.isfinite(f_offset):
                m_offset = np.nanmin(F_model)  # must be subtracted first to get close to zero
                return [f - m_offset + f_offset for f in F_model]
            else:
                return F_model

        def __len__(self):
            """should replace ``len(self.X)`` etc, not fully in use yet"""
            return len(self.fvalues)

        @property
        def evaluation_fraction(self):
            return self.evaluations / len(self.fvalues)

        @property
        def evaluations(self):
            return sum(self.evaluated)

        @property
        def remaining(self):
            """number of not yet evaluated solutions"""
            return len(self.X) - sum(self.evaluated)

    def __call__(self, X):
        self.count += 1
        model = self.model  # convenience shortcut
        # a trick to see whether the population size has increased (from a restart)
        # model.max_absolute_size is by default initialized with zero
        # TODO: remove dependency on the initial value of model.max_absolute_size
        if self.settings.model_max_size_factor * len(X) > model.settings.max_absolute_size:
            if model.settings.max_absolute_size:  # do not reset in first call, in case model was initialized meaningfully
                model.reset()  # reset, because the population size changed
            model.settings.max_absolute_size = self.settings.model_max_size_factor * len(X)
        evals = SurrogatePopulation.EvaluationManager(X)
        self.evals = evals  # only for the record

        if 11 < 3:
            # make minimum_model_size unconditional evals in the first call and quit
            if model.size < self.settings.minimum_model_size:
                evals.eval_sequence(self.settings.minimum_model_size - model.size,
                                    self.fitness, model.add_data_row)
                self.evaluations += evals.evaluations
                model.sort(self.settings.model_sort_globally or evals.evaluations)
                return evals.surrogate_values(model.eval, self.settings.return_true_fitnesses)

        if 11 < 3 and self.count % (1 + self.settings.crazy_sloppy):
            return evals.surrogate_values(model.eval, f_offset=model.F[0])

        number_evaluated = int(1 + max((len(X) * self.settings.min_evals_percent / 100,
                                        3 / model.settings.truncation_ratio - model.size)))
        while evals.remaining:
            idx = np.argsort([model.eval(x) for x in X]) if model.size > 1 else None
            yield from evals.eval_sequence(number_evaluated, self.fitness,
                                           model.add_data_row, idx)
            model.sort(
                number_evaluated)  # makes only a difference if elements of X are pushed out on later adds in evals
            tau = model.kendall(self.settings.n_for_tau(len(X), evals.evaluations))
            if evals.last_evaluations == number_evaluated:  # first call to evals.eval_sequence
                self.logger.add(tau)  # log first tau
            if tau >= self.settings.tau_truth_threshold - self.settings.tau_truth_threshold_correction * evals.evaluation_fraction:
                break
            number_evaluated += int(np.ceil(number_evaluated / 2))
            """multiply with 1.5 and take ceil
                [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 315, 473]
                +[1, 1, 2, 3, 4,  6,  9, 14, 21, 31, 47,  70, 105, 158]
            """

        model.sort(self.settings.model_sort_globally or evals.evaluations)
        model.adapt_max_relative_size(tau)
        self.logger.add(tau)  # log last tau
        self.evaluations += evals.evaluations
        if self.evaluations == 0:  # can currently not happen
            # a hack to have some grasp on zero evaluations from outside
            self.evaluations = 1e-2  # hundred zero=iterations sum to one evaluation
        self.logger.add(evals.evaluations / len(X))
        self.logger.push()
        return evals.surrogate_values(model.eval, self.settings.return_true_fitnesses)
