import matplotlib.pyplot as plt
import numpy as np
from ezmodel.core.factory import models_from_clazzes
from ezmodel.core.selection import ModelSelection
from ezmodel.models.kriging import Kriging
from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.util.display import Display
from pymoo.util.normalization import ZeroToOneNormalization
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
from pysamoo.experimental.acquisition import EI, AcquisitionProblem


# ---------------------------------------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------------------------------------


class EGODisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("f_opt", algorithm.opt[0].F[0])

        if algorithm.acq is None:
            self.output.append("f_new", "-")
            self.output.append("acq", "-")

            if algorithm.model_selection:
                self.output.append("regr", "-")
                self.output.append("corr", "-")
                self.output.append("ARD", "-")
        else:
            self.output.append("f_new", algorithm.infills.get("F").min())
            self.output.append("acq", - algorithm.infills[0].get("acq"))

            if algorithm.model_selection:
                gp = algorithm.surrogate
                self.output.append("regr", gp.regr)
                self.output.append("corr", gp.corr)
                self.output.append("ARD", gp.ARD)


# ---------------------------------------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------------------------------------


class BayesianOptimization(SurrogateAssistedAlgorithm):

    def __init__(self,
                 acq_func=EI(),
                 model_selection=False,
                 adaptive_fmin=True,
                 display=EGODisplay(),
                 **kwargs):

        super().__init__(display=display, **kwargs)
        self.default_termination = SingleObjectiveDefaultTermination()

        self.model_selection = model_selection
        self.acq_func = acq_func
        self.adaptive_fmin = adaptive_fmin
        self.acq = None

    def _infill(self):

        # get all the points that have been evaluated yet
        X, F = self.archive.get("X", "F")

        # get he problem and the boundaries
        problem = self.problem
        xl, xu = problem.bounds()

        # the defaults for surrogate modeling - normalize the values to be between zero and one
        defaults = dict(norm_X=ZeroToOneNormalization(xl, xu))

        # rather the best model should be selected or simply the default kriging implementation taken
        if self.model_selection:
            models = models_from_clazzes(Kriging, **defaults)
            partitions = CrossvalidationPartitioning(k_folds=5, seed=1).do(X)
            model = ModelSelection(models).do(X, F[:, 0], partitions)
        else:
            model = Kriging(regr="linear", corr="gauss", ARD=True, **defaults)
            model.fit(X, F[:, 0])

        if self.adaptive_fmin:

            # get the acquisition problem to be optimized
            acq = robust_fmin_acquisition(problem, model, self.acq_func, self.archive)

        else:
            # just use the minimum (even though this can lead to precision issues)
            _min = F[:, 0].argmin()
            f_min = model.predict(X[_min])[0, 0]
            acq = AcquisitionProblem(problem, model, self.acq_func, f_min=f_min)

        # use a bigger latin hypercube in the beginning because the problem might be highly multi-modal
        sampling = LHS().do(problem, 500)

        algorithm = NicheGA(pop_size=50, sampling=sampling)

        termination = SingleObjectiveDefaultTermination(nth_gen=1, n_last=5)

        res = minimize(acq,
                       algorithm,
                       termination,
                       verbose=False
                       )

        X, F = res.opt.get("X", "F")

        self.acq = acq
        self.surrogate = model
        return Population.new(X=X, acq=F[0])

    def _set_optimum(self):
        self.opt = FitnessSurvival().do(self.problem, self.archive, n_survive=1)


def robust_fmin_acquisition(problem, model, acq_func, points):
    X, F = points.get("X", "F")

    sorted_by_pred = np.sort(model.predict(X)[:, 0])

    n_intervals = min(20, len(X))
    n_points = 500

    interval = int(len(sorted_by_pred) / n_intervals)

    for cnt in range(n_intervals):

        # increase the index for f_min in each iteration
        f_min = sorted_by_pred[cnt * interval]

        # create the acquisition problem and radomly sample
        acq = AcquisitionProblem(problem, model, acq_func, f_min=f_min)
        sampling = LHS().do(problem, n_points)

        # the maximum value found of during random sampling
        max_prob_imprv = (- acq.evaluate(sampling.get("X"))).max()

        # print(cnt, sorted_by_pred[0], f_min, max_prob_imprv)

        # if the value is not very small then the fmin is okay to be used
        if max_prob_imprv > 1e-2:
            break

    return acq


class EGOVisualization(Callback):

    def notify(self, algorithm):
        problem = algorithm.problem
        if problem.n_var > 1 or problem.n_obj > 1 or algorithm.surrogate is None:
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

        gp = algorithm.surrogate
        mu, sigma = gp.predict(mesh, return_values_of=["y", "sigma"])

        plt_func.fill_between(mesh[:, 0], (mu - 2 * sigma)[:, 0], (mu + 2 * sigma)[:, 0], alpha=0.2, color='k')

        plt_func.scatter(infill.X, infill.F, color="red", s=100, marker="x")
        plt_func.plot(mesh, gp.predict(mesh), color="red")

        plt_func.axvline(x=algorithm.infills[0].X, color="black", linestyle='dashed')

        plt_func.plot(mesh, problem.evaluate(mesh), color="black")

        plt_acq.plot(mesh, acq.evaluate(mesh), color="blue")
        plt_acq.scatter(infill.X, acq.evaluate(infill.X), color="red", s=100, marker="x")

        plt.show()
