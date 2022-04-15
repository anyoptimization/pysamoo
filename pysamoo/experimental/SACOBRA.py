import numpy as np
import scipy

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import random
from pymoo.optimize import minimize
from pymoo.problems.meta import MetaProblem
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import cdist
from pymoo.util.normalization import NoNormalization
from pymoo.util.optimum import filter_optimum
from pysamoo.core.surrogate import Surrogate
from ezmodel.models.rbf import RBF
from ezmodel.util.transformation.plog import Plog
from ezmodel.util.transformation.zero_to_one import ZeroToOneNormalization


# =========================================================================================================
# Display
# =========================================================================================================


class SACOBRADisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("eps", algorithm.eps)
        self.output.append("rho", algorithm.rho if algorithm.rho is not None else "-")
        self.output.append("C_feas", int(algorithm.C_feas))
        self.output.append("C_infeas", int(algorithm.C_infeas))


class SACOBRA(Algorithm):

    def __init__(self, n_doe=None, display=SACOBRADisplay(), **kwargs):
        super().__init__(display=display, **kwargs)
        self.n_doe = n_doe
        self.archive = Population()

        self.rho = None

        self.C_feas = 0
        self.C_infeas = 0

        self.T_feas = None
        self.T_infeas = None

        self.repair = None

    def _setup(self, problem, **kwargs):

        # directly rescale the whole problem
        problem = Rescale(problem)

        if self.n_doe is None:
            self.n_doe = 2 * problem.n_var + 1

        xl, xu = problem.bounds()
        l = (xu - xl).min()
        self.eps = 0.005 * l
        self.eps_max = 0.01 * l

        self.T_feas = np.floor(2 * np.sqrt(problem.n_var))
        self.T_infeas = self.T_feas

        #
        models = {}

        for label, norm in [("default", NoNormalization()), ("plog", Plog())]:
            model = RBF(kernel='cubic', tail='linear', normalized=False, norm_y=norm)
            models[f"rbf-{label}"] = model

        targets = {
            "F": ObjectivesAsTarget({k: v for (k, v) in models.items() if "plog" not in k}),
            "G": ConstraintsAsTarget(models),
        }

        self.surrogate = Surrogate(problem, targets)

        self.problem = problem

    def _initialize_infill(self):
        return LHS().do(self.problem, self.n_doe)

    def _initialize_advance(self, infills=None, **kwargs):
        self.archive = Population.merge(self.archive, infills)

        # analyze the initial population and find ranges
        self._analyze_initial_pop(infills)

        # adjust the constraint values
        self._adjust_constr(infills)

        # adjust the DRC parameter based on whether the obj. function is steep or not
        self._adjust_drc()

        # initialize the surrogate with the doe points
        self.surrogate.initialize(infills)

    def _infill(self):

        # update the current surrogate(s) based on the most recent update done after the last generation
        self.surrogate.advance()

        # and now finally create the models
        self.surrogate.fit(metric="spear", minimize=False)

        verbose = False

        if verbose:

            for target, entry in self.surrogate.targets["F"].predictor.predictors.items():
                print(target, entry["label"], entry["mae"])

            for target, entry in self.surrogate.targets["G"].predictor.predictors.items():
                print(target, entry["label"], entry["mae"])

        # get the current best solution and the infill
        feas = self.archive.get("feas")
        opt, sol = self.opt[0], self.archive[-1]

        # if the new solution has a better objective value than the current one, flag the next iteration for repair
        if not np.any(feas):

            x_new = self._repair()

        elif not sol.feas and sol.f < opt.f:

            # attempt to repair the promising solution
            x_new = self._repair(sol)

        # this is the regular iteration when obj. and constr. are used for optimization
        else:

            # find the rho by cycling through the XI array
            self.rho = self.XI[(self.n_gen - 1) % len(self.XI)]

            x_start = self._random_restart()

            # search on the surrogate (cobra iteration)
            x_new = self._cobra(x_start, self.eps, self.rho)

        return Population.new(X=[x_new])

    def _advance(self, infills=None, **kwargs):

        # adjust the constraint values
        self._adjust_constr(infills)

        # adjust the eps value (margin for infeasibility)
        self._adjust_margins(infills)

        # add them to the archive
        self.archive = Population.merge(self.archive, infills)

        # add the new infill solutions to the surrogate to be used in the next iteration
        self.surrogate.update(infills)

    def _analyze_initial_pop(self, pop):
        f, G = pop.get("f", "G")
        self.FR = f.max() - f.min()
        self.GR = G.max(axis=0) - G.min(axis=0)

    def _adjust_constr(self, pop):
        G = pop.get("G")
        _G = G * (self.GR.mean() / self.GR)
        pop.set("G", _G)

    def _adjust_margins(self, infills):
        sol = infills[0]

        if sol.feas:
            self.C_feas += 1
            self.C_infeas = 0
        else:
            self.C_feas = 0
            self.C_infeas += 1

        if self.C_feas >= self.T_feas:
            self.eps = 0.5 * self.eps
            self.C_feas = 0

        if self.C_infeas >= self.T_infeas:
            self.eps = min(2 * self.eps, self.eps_max)
            self.C_infeas = 0

    def _cobra(self, x_start, eps, rho):

        X = self.archive.get("X")

        tcv = TotalConstraintViolation(ieq_eps=-1 * eps)

        def func_obj(x):
            return self.surrogate.evaluate(x, return_values_of=["F"])[0]

        def func_constr(x):
            G = self.surrogate.evaluate(x, return_values_of=["G"])
            cv = tcv.calc(G)
            return -1 * (cv[0])

        def func_constr_trust(x):
            closest = cdist(x[None, :], X).min()
            trust = rho - closest
            return -1 * trust

        constraints = [{"type": 'ineq', "fun": func_constr}, {"type": 'ineq', "fun": func_constr_trust}]

        xl, xu = self.problem.bounds()
        bounds = np.column_stack([xl, xu])

        options = {}

        res = scipy.optimize.minimize(func_obj,
                                      x_start,
                                      args=(),
                                      bounds=bounds,
                                      method='SLSQP',
                                      constraints=constraints,
                                      tol=None,
                                      callback=None,
                                      options=options)

        print(func_obj(res.x), func_constr(res.x), func_constr_trust(res.x))
        print(res.x)
        return res.x

    def _random_restart(self):
        p1 = 0.125
        p2 = 0.4

        feas = self.archive.get("feas")

        if feas.sum() / len(feas) < 0.05:
            p = p2
        else:
            p = p1

        if np.random.random() < p:
            x = random(self.problem)[0]
        else:
            x = self.opt[0].X

        return x

    def _repair(self, sol=None):

        tcv = TotalConstraintViolation()

        class MyProblem(MetaProblem):

            def _evaluate(self, x, out, *args, **kwargs):
                super()._evaluate(x, out, *args, **kwargs)

                if sol is not None:
                    out["F"] = cdist(x, sol.X[None, :])
                else:
                    out["F"] = tcv.calc(out["G"])
                    out["G"][:] = 0.0

        problem = MyProblem(self.surrogate)

        sampling = Population.new(X=self.archive.get("X"))
        self.evaluator.eval(problem, sampling, count_evals=False)

        # let us start with a maximum of 100 solutions
        if len(sampling) > 100:
            sampling = FitnessSurvival().do(self.surrogate, sampling, n_survive=100)

        algorithm = DE(pop_size=len(sampling), sampling=sampling)

        res = minimize(problem,
                       algorithm,
                       return_least_infeasible=True)

        return res.X

    def _adjust_drc(self):
        if self.FR > 1000:
            self.XI = np.array([0.001, 1e-16])
        else:
            self.XI = np.array([0.3, 0.05, 0.001, 0.0005, 1e-16])

    def _set_optimum(self):
        self.opt = filter_optimum(self.archive, least_infeasible=True)


class MinusOneToOneNormalization(ZeroToOneNormalization):

    def forward(self, X):
        return super().forward(X) * 2 - 1

    def backward(self, X):
        return super().backward((X + 1) / 2)


class Rescale(MetaProblem):

    def __init__(self, problem):
        super().__init__(problem)
        assert self.xl is not None and self.xu is not None, "Both, xl and xu, must be set to redefine the problem!"

        self.norm = MinusOneToOneNormalization(problem.xl, problem.xu)
        self.xl, self.xu = -np.ones(self.n_var), np.ones(self.n_var)

    def do(self, x, out, *args, **kwargs):
        out["__X__"] = x
        xp = self.norm.backward(x)
        super().do(xp, out, *args, **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        ps = super()._calc_pareto_set(*args, **kwargs)
        return self.norm.forward(ps)
