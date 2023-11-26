from copy import deepcopy

import numpy as np
from ezmodel.core.factory import models_from_clazzes
from ezmodel.models.knn import KNN
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.util.display.column import Column
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.dominator import get_relation
from pymoo.util.misc import norm_eucl_dist
from pymoo.util.normalization import ZeroToOneNormalization

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
from pysamoo.core.surrogate import Surrogate
from pysamoo.core.target import Target


class PSAFOutput(SingleObjectiveOutput):

    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)
        self.output = output

        self.bias = Column(name="bias", func=lambda a: a.bias)
        self.r2 = Column(name="r2", func=lambda a: a.r2)
        self.only_one_mode = False
        self.mae = Column(name="mae")
        self.model = Column(name="model",  width=60)

    def initialize(self, algorithm):
        self.output.initialize(algorithm)
        self.columns = [e for e in self.output.columns if e.name != 'f_avg'] + [self.r2, self.bias]

        problem = algorithm.problem
        if problem.n_obj == 1 and problem.n_constr == 0:
            self.only_one_mode = True
            self.columns += [self.mae, self.model]

    def update(self, algorithm):
        psaf = algorithm
        self.output.update(psaf)
        self.bias.update(psaf)
        self.r2.update(psaf)

        if self.only_one_mode:
            target = psaf.surrogate.targets[0]
            self.mae.set(target.performance("mae"))
            self.model.set(target.best)


class PSAF(SurrogateAssistedAlgorithm):

    def __init__(self, algorithm, alpha=5, beta=30, rho=None, rho_max=0.7, eps=0.005, **kwargs):
        SurrogateAssistedAlgorithm.__init__(self, **kwargs)
        self.algorithm = deepcopy(algorithm)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.rho = rho
        self.rho_max = rho_max
        self.bias = rho
        self.r2 = None

    def _setup(self, problem, **kwargs):
        assert problem.n_obj == 1 and problem.n_constr == 0, "PSAF only works for unconstrained single-objective problems!"

        super()._setup(problem, **kwargs)
        self.algorithm.setup(problem, **kwargs)

        # customize the display to show the surrogate influence
        self.display = self.algorithm.display
        self.display.output = PSAFOutput(self.algorithm.output)
        self.algorithm.display = lambda _: None

        xl, xu = problem.bounds()
        defaults = dict(norm_X=ZeroToOneNormalization(xl, xu))

        models = models_from_clazzes(RBF, **defaults)
        models = {name: entry["model"] for name, entry in models.items()}
        models = {**models, **{"krg-cont": Kriging(regr="constant"), "krg-lin": Kriging(regr="linear")}}

        if "baseline" not in models:
            models["baseline"] = KNN(problem.n_var + 1)

        # create the ensemble of surrogates considering objectives and constraints
        targets = [Target(("F", 0), models)]
        self.surrogate = Surrogate(problem, targets)

        # define the survival for the individuals to keep
        self.survival = FitnessSurvival() if self.problem.n_obj == 1 else RankAndCrowdingSurvival()

    def _initialize_advance(self, infills=None, **kwargs):

        # validate and check different surrogates to find the best
        self.surrogate.validate(infills, exclude=["baseline"])

        # now we perform a fake initialization of the algorithm object by providing individuals from LHS
        fake = self.algorithm.infill()
        fittest = self.survival.do(self.problem, infills, n_survive=len(fake))
        self.algorithm.advance(infills=fittest)

        super()._initialize_advance(infills=infills, **kwargs)

    def _infill(self):
        problem, algorithm, surrogate = self.surrogate.problem(), self.algorithm, self.surrogate

        # advance the surrogate which results in doing a benchmark and setting the best surrogate combination
        surrogate.fit(self._archive)

        # set the bias the surrogate is supposed to have - only if greater than zero we do the second phase
        target = surrogate.targets[0]
        r2 = 1 - (target.performance("mse") / target.performance("mse", model="baseline"))
        bias = max(self.rho_max, r2) if self.rho is None else self.rho

        # calculate the infill solutions as the algorithms usually would
        off = algorithm.infill()

        # use the ensemble to values the current offsprings
        Evaluator().eval(problem, off)

        # if a tournament selection should be done alpha is at least two
        if self.alpha > 1:

            # do the tournament for each alpha
            for k in range(self.alpha - 1):

                # create a second pool and actually do the tournament
                others = self.algorithm.infill()
                Evaluator().eval(problem, others)

                # for each offspring see if we do the surrogate tournament
                for k in range(len(off)):

                    # if the competitor is not worse it will take the lead
                    if get_relation(others[k], off[k]) >= 0:
                        off[k] = others[k]

        # if algorithm shall be continued on the surrogate and there is a bias at all
        if self.beta > 0 and bias > 0.0:

            # already calculate what individuals will be replaced later
            replace = np.random.random(len(off)) <= bias

            # if at least one is replaced actually simulate the algorithm on the surrogate
            if replace.sum() > 0:

                # create a copy of the algorithm object
                algorithm = deepcopy(self.algorithm)
                algorithm.termination = NoTermination()
                Evaluator().eval(problem, algorithm.pop, skip_already_evaluated=False)

                # the candidates to take from beta are initialized to the offsprings
                cands = off.copy()

                # run the algorithm for beta generations and always assign replacement candidates
                for k in range(self.beta):

                    # just to make sure no algorithm specific termination criterion has be executed
                    if not algorithm.has_next():
                        break

                    # get the infills from the algorithm and evaluate on the surrogate
                    infills = algorithm.infill()
                    Evaluator().eval(problem, infills)

                    # if there is some candidates to consider
                    if len(infills) > 0:

                        # find the closest individuals for each candidate to offsprings
                        dists = norm_eucl_dist(problem, infills.get("X"), off.get("X"))
                        I = dists.argmin(axis=1)

                        # for each infill solution check if it replaces the candidate
                        for j in range(len(infills)):

                            # get the index to the closest of offsprings
                            i = I[j]

                            # if it is better indeed then we update it
                            if get_relation(infills[j], cands[i]) >= 0:
                                cands[i] = infills[j]

                    # now continue the execution of the algorithm
                    algorithm.advance(infills=infills)

                # now do the probabilistic replacement
                for i in range(len(off)):

                    # if it should be replaced (that has been pre-calculated)
                    if replace[i]:
                        off[i] = cands[i]

        self.bias = bias
        self.r2  = r2

        # no only use the X values
        infills = Population.new(X=off.get("X"))

        # set the velocity for PSO
        infills.set("V", off.get("V"), "index", off.get("index"))

        return infills

    def _advance(self, infills=None, **kwargs):
        # update the surrogate(s) with the new infills points
        self.surrogate.validate(self._archive, infills, exclude=["baseline"])

        # make a step in the main algorithm with high-fidelity solutions
        self.algorithm.advance(infills=infills, **kwargs)

        super()._advance(infills=infills, **kwargs)

    def _set_optimum(self):
        self.opt = self.algorithm.opt
