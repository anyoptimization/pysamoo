import numpy as np

try:
    from smac.facade.func_facade import fmin_smac
except:
    raise Exception("smac not found. Please execute: 'pip install smac'")



from pymoo.algorithms.base.local import LocalSearch
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.optimum import filter_optimum


class FunctionCall:

    def __init__(self, problem) -> None:
        super().__init__()
        self.problem = problem

    def __call__(self, x):
        return self.problem.evaluate(x[None, :])[0, 0]


class SMAC(LocalSearch):

    def __init__(self, x0=None,
                 sampling=LatinHypercubeSampling(),
                 display=SingleObjectiveDisplay(),
                 n_sample_points="auto",
                 n_max_sample_points=50,
                 **kwargs):

        super().__init__(x0, sampling, display, n_sample_points, n_max_sample_points, **kwargs)

        self.cnt = 0
        self.history = None

    def _local_advance(self, **kwargs):

        if self.history is None:
            self.history = self._optimize()

        self.evaluator.n_eval += 1
        self.pop = self.history.pop(0)
        self.cnt += 1

        if len(self.history) == 0:
            self.termination.force_termination = True

    def _optimize(self):

        problem = self.problem

        func = FunctionCall(problem)

        seed = np.random.randint(0, 2 ** 32 - 1)

        eval_remaininig = self.termination.n_max_evals - self.evaluator.n_eval

        x, cost, obj = fmin_smac(func=func,
                                 x0=self.x0.X,
                                 bounds=np.column_stack(problem.bounds()),
                                 maxfun=eval_remaininig,
                                 rng=seed,
                                 scenario_args=dict(output_dir=None))

        pop = []

        for k, v in obj.runhistory.data.items():
            config_id = k.config_id
            f = v.cost
            x = np.array(list(obj.runhistory.ids_config[config_id]._values.values()))
            ind = Individual(X=x, F=np.array([f]), CV=np.array([0.0]), feasible=[True])
            pop.append(Population.create(ind))

        return pop

    def _set_optimum(self):
        pop = self.pop if self.opt is None else Population.merge(self.opt, self.pop)
        self.opt = filter_optimum(pop, least_infeasible=True)


