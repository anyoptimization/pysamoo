from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.problems.meta import MetaProblem
from pymoo.util.reference_direction import select_points_with_maximum_distance


class NichingConstrainedSampling(Sampling):

    def __init__(self, func_eval_constr, sampling=LHS(), initial_eps=0.25):
        super().__init__()
        self.func_eval_constr = func_eval_constr
        self.initial_eps = initial_eps
        self.sampling = sampling

    def _do(self, problem, n_samples, **kwargs):
        constr = self.func_eval_constr

        class ConstrainedProblem(MetaProblem):

            def __init__(self, problem):
                super().__init__(problem)
                self.n_obj = 1
                self.n_constr = 1

            def _evaluate(self, x, out, *args, **kwargs):
                cv = constr(x)
                out["F"] = cv
                out["G"] = cv

        problem = ConstrainedProblem(problem)

        eps = self.initial_eps

        while True:

            algorithm = NicheGA(pop_size=n_samples, samping=self.sampling, norm_niche_size=eps, norm_by_dim=True)

            res = minimize(problem, algorithm, ("n_gen", 200), return_least_infeasible=True)
            opt = res.opt

            X = opt.get("X")[opt.get("CV")[:, 0] <= 0]

            print(eps, len(X))

            if len(X) >= n_samples:
                break

            eps = eps * 0.9

        if len(X) > n_samples:
            I = select_points_with_maximum_distance(X, n_samples)
            X = X[I]

        return X
