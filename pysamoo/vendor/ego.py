import numpy as np

try:
    from GPyOpt.methods import BayesianOptimization
except:
    raise Exception("gpyopt not found. Please execute: 'pip install gpyopt'")

from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.display import SingleObjectiveDisplay


class EGO(Algorithm):

    def __init__(self,
                 display=SingleObjectiveDisplay(),
                 **kwargs):
        super().__init__(display=display, **kwargs)
        self.domain, self.func = None, None

    def _setup(self, problem, **kwargs):

        domain = []
        for k in range(problem.n_var):
            e = {'name': f"var_{k}", 'type': 'continuous', 'domain': (problem.xl[k], problem.xu[k])}
            domain.append(e)

        def func(x):
            ind = Individual(X=x[0])
            self.evaluator.eval(self.problem, ind)
            self.pop = Population.merge(self.pop, Population.create(ind))
            return ind.F[0]

        self.obj = BayesianOptimization(f=func, domain=domain)

    def _initialize_advance(self, **kwargs):
        self._advance()

    def _advance(self, **kwargs):
        obj = self.obj

        try:
            obj._update_model(obj.normalization_type)
        except np.linalg.linalg.LinAlgError:
            pass

        obj.num_acquisitions += 1
        obj.suggested_sample = obj._compute_next_evaluations()
        obj.X = np.vstack((obj.X, obj.suggested_sample))
        obj.evaluate_objective()

