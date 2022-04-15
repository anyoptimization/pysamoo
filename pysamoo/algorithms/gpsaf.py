import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.tournament import compare
from pymoo.optimize import default_termination
from pymoo.util.display import Display
from pymoo.util.dominator import get_relation
from pymoo.util.misc import norm_eucl_dist, cdist
from pymoo.util.optimum import filter_optimum
from pymoo.util.termination.no_termination import NoTermination
from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.visualization.video.callback_video import AnimationCallback

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
# =========================================================================================================
# Display
# =========================================================================================================
from pysamoo.core.knockout import noisy


class GPSAFDisplay(Display):

    def __init__(self, display, **kwargs):
        super().__init__(**kwargs)
        self.display = display

    def _do(self, problem, evaluator, gpasf):
        self.display.do(problem, evaluator, gpasf.algorithm, show=False)
        self.output = self.display.output

        if gpasf.n_gen > 1:
            surr_infills = Population.create(*gpasf.infills.get("created_by"))
            n_influenced = sum(surr_infills.get("type") == "trace")
            self.output.append("n_influenced", f"{n_influenced}/{len(surr_infills)}")
        else:
            self.output.append("n_influenced", "-")

        if problem.n_obj == 1 and problem.n_constr == 0:
            if len(gpasf.surrogate.targets) >= 1:
                target = gpasf.surrogate.targets[0]
                self.output.append("mae", target.performance("mae"))
                self.output.append("model", target.best)

        elif problem.n_obj == 2 and problem.n_constr == 0:
            if len(gpasf.surrogate.targets) >= 2:
                perf = gpasf.surrogate.performance("mae")
                if ("F", 0) in perf:
                    self.output.append("mae f1", perf[("F", 0)])
                if ("F", 1) in perf:
                    self.output.append("mae f2", perf[("F", 1)])


# =========================================================================================================
# Animation
# =========================================================================================================

class GPSAFAnimation(AnimationCallback):

    def __init__(self,
                 nth_gen=1,
                 n_samples_for_surface=200,
                 dpi=200,
                 **kwargs):

        super().__init__(nth_gen=nth_gen, dpi=dpi, **kwargs)
        self.n_samples_for_surface = n_samples_for_surface
        self.last_pop = None

    def do(self, problem, algorithm):

        if problem.n_var != 2 or problem.n_obj != 1:
            raise Exception(
                "This visualization can only be used for problems with two variables and one objective!")

        # draw the problem surface
        doe = algorithm.surrogate.targets["F"].doe
        if doe is not None:
            problem = algorithm.surrogate

        plot = FitnessLandscape(problem, _type="contour", kwargs_contour=dict(alpha=0.5))
        plot.do()

        if doe is not None:
            plt.scatter(doe.get("X")[:, 0], doe.get("X")[:, 1], color="black", alpha=0.3)

        for k, sols in enumerate(algorithm.trace_assigned):
            if len(sols) > 0:
                pop = Population.create(*sols)
                plt.scatter(pop.get("X")[:, 0], pop.get("X")[:, 1], color="blue", alpha=0.3)

                x = algorithm.influenced[k].X
                for sol in sols:
                    plt.plot((x[0], sol.X[0]), (x[1], sol.X[1]), alpha=0.1, color="black")

        plt.scatter(algorithm.influenced.get("X")[:, 0], algorithm.influenced.get("X")[:, 1], color="red", marker="*",
                    alpha=0.7,
                    label="influenced")

        _biased = Population.create(
            *[e for e in algorithm.biased if e is not None])
        plt.scatter(_biased.get("X")[:, 0], _biased.get("X")[:, 1], color="orange", marker="s", label="Selected",
                    alpha=0.8,
                    s=100)

        plt.legend()


# =========================================================================================================
# Algorithm
# =========================================================================================================


class GPSAF(SurrogateAssistedAlgorithm):

    def __init__(self,
                 algorithm,
                 alpha=5,
                 beta=30,
                 rho=None,
                 n_max_doe=100,
                 n_max_infills=np.inf,
                 **kwargs):

        SurrogateAssistedAlgorithm.__init__(self, **kwargs)

        self.proto = deepcopy(algorithm)
        self.algorithm = None

        # the control parameters for the surrogate assistance
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_max_doe = n_max_doe

        # the maximum number of infill solutions
        self.n_max_infills = n_max_infills

        self.surr_infills = None
        self.influenced, self.trace, self.biased = None, None, None
        self.restart = False

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)

        # set the default termination to the proto type
        if self.proto.termination is None:
            self.proto.termination = default_termination(problem)

        # setup the underlying algorithm
        self.proto.setup(problem, seed=self.seed, **kwargs)

        # copy the algorithm object to get started
        self.algorithm = deepcopy(self.proto)

        # customize the display to show the surrogate influence
        self.display = GPSAFDisplay(self.algorithm.display)

        # define the survival for the individuals to keep
        self.survival = FitnessSurvival() if problem.n_obj == 1 else RankAndCrowdingSurvival()

    def _initialize_infill(self):
        # return self.algorithm.infill()
        return super()._initialize_infill()

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)

        # validate and check different surrogates to find the best
        self.surrogate.validate(infills)

        # now we perform a fake initialization of the algorithm object by providing individuals from LHS
        # fake = self.algorithm.infill()
        # fittest = self.survival.do(self.problem, infills, n_survive=len(fake))
        fittest = infills

        # feed back the fittest individuals to the algorithm
        self.algorithm.advance(infills=fittest)

    def _infill(self):

        # if the algorithm should do a restart - copy again the prototype
        if self.restart:
            self.algorithm = deepcopy(self.proto)

        # get the design of experiments to be used for modeling
        doe = self._doe()

        # always before using the surrogate fit it
        self.surrogate.fit(doe)

        # get the error estimate from the surrogate
        error = self.surrogate.performance("mae")

        # this would be the default behavior of the algorithm
        influenced = self.algorithm.infill()

        # do the alpha phase based on the tournament selection based on the surrogate prediction
        if self.alpha is not None:
            influenced = self._infill_alpha(influenced, error=error)

        # now by default the infills are the surrogate-influenced solutions
        infills = influenced
        infills.set("type", "influenced")

        # continue running the algorithm for more generations if beta is "enabled"
        if self.beta > 0:

            # get the trace from the beta run on the surrogate
            trace = self._infill_beta()
            trace.set("type", "trace")
            self.trace = trace

            # 3) assign the found solutions to the original infill solutions
            trace_assigned = self._infill_beta_assign(influenced, trace)

            # when a solution is considered a duplicate
            eps = 1e-12

            # for each solution from the surrogate based optimization
            for i, pool in enumerate(trace_assigned):

                # if there are no solutions to replace continue directly
                if len(pool) == 0:
                    continue

                # create a population object from the pool
                pool = Population.create(*pool)

                # the probability of replacement
                rho = self.rho

                # if not fixed we use the distribution of points
                if rho is None:
                    rho = (len(pool) / max([len(e) for e in trace_assigned])) ** 0.5

                # if the solution should be replaced
                if np.random.random() <= rho:

                    # if it should be replaced find the ONE solution from the pool
                    biased = self._infill_prob_tourn(pool, method="tournament", error=error, n_winners=1)[0]

                    # check the distance to existing solutions
                    closest = norm_eucl_dist(self.problem, biased.get("X"), self.archive.get("X")).min()

                    # if the solution is in fact new
                    if closest > eps:

                        # now actually set the value to the infills
                        infills[i] = biased
                        # infills[i].X = biased.X

                    else:
                        print("BIASED: TOO CLOSE (SKIP)")

                closest = norm_eucl_dist(self.problem, infills[i].get("X"), self.archive.get("X")).min()

                # if the solution is in fact new
                if closest < eps:
                    print("INFLUENCED: TOO CLOSE")

        # if beta is zero, then simply take the results from the alpha phase
        else:
            infills = influenced

        # now copy over the infills and set them to have never been evaluated
        ret = infills.copy(deep=True)
        for e in ret:
            e.reset(data=False)
        ret.set("X", infills.get("X"), "created_by", infills)

        return ret

    def _infill_alpha(self, infills, error=None):
        problem = self.surrogate.problem()

        # create the influenced population using the tournament
        influenced = infills

        # use the ensemble to values the current offsprings
        Evaluator().eval(problem, influenced)

        # reduce the number of infill solutions if required
        if len(influenced) > self.n_max_infills:
            influenced = self._infill_prob_tourn(influenced, method="tournament", error=error, n_winners=self.n_max_infills)

        # do the tournament for each alpha
        for k in range(self.alpha - 1):

            # create a second pool and actually do the tournament
            others = self.algorithm.infill()
            if len(others) > len(influenced):
                I = np.random.permutation(len(others))[:len(influenced)]
                others = others[I]

            Evaluator().eval(problem, others)

            # for each offspring see if we do the surrogate tournament
            for k in range(len(influenced)):

                # if the competitor is not worse it will take the lead
                if get_relation(influenced[k], others[k]) < 1:
                    influenced[k] = others[k]

        return influenced

    def _infill_beta_assign(self, influenced, trace, filter=False):
        ret = [[] for _ in range(len(influenced))]

        if len(trace) > 0:

            # find the closest individuals for each candidate to offsprings (and NOT the current replaced one)
            dists = norm_eucl_dist(self.problem, trace.get("X"), influenced.get("X"))
            closest = dists.argmin(axis=1)

            for k, i in enumerate(closest):

                # add the solution closest to the influence to the list
                if not filter or get_relation(influenced[i], trace[k]) < 1:
                    trace[k].set("i", i)
                    ret[i].append(trace[k])

        return ret

    def _infill_beta(self):

        # define the problem on the surrogate
        problem = self.surrogate.problem()

        # create a copy of the algorithm object to keep the original unmodified
        algorithm = deepcopy(self.algorithm)
        # if np.random.random() < 0.5:
        #     algorithm = deepcopy(self.algorithm)
        # else:
        #     algorithm = deepcopy(self.proto)
        #     print("PROTO")

        # disable the termination to have have enough iterations to continue
        algorithm.termination = NoTermination()
        algorithm.has_terminated = False

        # now use the surrogate to evaluate the solutions from the current algorithm object
        if algorithm.pop is not None:
            Evaluator().eval(problem, algorithm.pop, skip_already_evaluated=False)

        trace = []

        # run the algorithm for beta iterations
        for k in range(self.beta):

            # just to make sure no algorithm specific termination criterion has be executed
            if not algorithm.has_next():
                break

            # get the infills from the algorithm and evaluate on the surrogate
            infills = algorithm.infill()
            infills.set("k", k)
            Evaluator().eval(problem, infills)

            trace.extend(infills)

            # now continue the execution of the algorithm
            algorithm.advance(infills=infills)

        return Population.create(*trace)

    def _infill_prob_tourn(self, sols, n_winners=1, method="tournament", error=None):

        # if the beta phase has not found any solutions close to the influenced one
        if len(sols) == 0:
            return None

        else:

            if method == "best":
                return FitnessSurvival().do(self.problem, Population.create(*sols), n_survive=n_winners)[0]

            elif method == "random":
                return np.random.choice(sols, size=n_winners)

            elif method == "tournament":

                # create a copy of all solutions to be considered
                pool = list(range(len(sols)))

                # until we have found a clear winner of the tournament
                while True:

                    # always shuffle the pool to have random tournaments
                    random.shuffle(pool)

                    # make sure the pool is an even number
                    if len(pool) % 2 != 0:
                        pool.append(np.random.choice(pool))

                    # create the pairs that compete with each other
                    pairs = np.reshape(np.array(pool), (-1, 2))

                    # prepare the next pool already containing all the winners
                    winners = []

                    # create a solution pool with noise
                    sols_with_noise = noisy(sols, error)

                    for i, j in pairs:

                        # the two solutions to be compared
                        a, b = sols_with_noise[i], sols_with_noise[j]

                        # calc the relation in a probabilistic manner
                        rel = get_relation(a, b)

                        if rel == 1:
                            winners.append(i)
                        elif rel == -1:
                            winners.append(j)
                        else:
                            if a.get("k") is not None and b.get("k") is not None:
                                k = compare(i, a.get("k"), j, b.get("k"), method='larger_is_better',
                                            return_random_if_equal=True)
                            else:
                                k = np.random.choice([i, j])

                            winners.append(k)

                    if len(winners) <= n_winners:

                        if len(winners) < n_winners:
                            H = set(winners)

                            random.shuffle(pool)

                            for k in pool:

                                if k not in H:
                                    winners.append(k)
                                    H.add(k)

                                if len(winners) == n_winners:
                                    break

                        return sols[winners]

                    pool = winners

            else:
                raise Exception("Unknown selection.")

    def _advance(self, infills=None, **kwargs):

        if self.restart:

            for k in np.random.permutation(len(infills)):

                opt = np.random.choice(self.opt)
                if get_relation(opt, infills[k]) >= 0:
                    infills[k] = opt
                    break

            self.restart = False

        # validate the current model
        self.surrogate.validate(trn=self.doe, tst=infills)

        # make a step in the main algorithm with high-fidelity solutions
        self.algorithm.advance(infills=infills, **kwargs)

        if not self.algorithm.has_next():
            # self.restart = True
            # print("RESTART")

            self.restart = False
            print("RESTART is DISABLED")

        # for target in self.surrogate.targets:
        #     print(target.label, target.best)


        super()._advance(infills=infills, **kwargs)

    def _doe(self):
        n_max_doe = self.n_max_doe

        doe = self.archive

        if len(doe) > n_max_doe:

            center = LHS().do(self.problem, n_max_doe).get("X")
            A = cdist(doe.get("X"), center).argmin(axis=1)

            cluster = [[] for _ in range(n_max_doe)]
            for i, a in enumerate(A):
                cluster[a].append(i)

            doe = []
            for c in cluster:
                if len(c) > 0:
                    s = np.random.choice(c)
                    doe.append(s)

            doe = self.archive[doe]
            doe = Population.merge(doe, self.algorithm.pop)
            doe = Population.merge(doe, self.algorithm.opt)
            doe = Population.merge(doe, self.opt)
            doe = DefaultDuplicateElimination().do(doe)

            # if we have constraints also select the closest infeas. or feas. solution
            others = []
            if self.problem.has_constraints():

                is_feas = self.archive.get("feas")
                feas, infeas = np.where(is_feas)[0], np.where(~is_feas)[0]

                for sol in doe:
                    if sol.get("feas"):
                        if len(infeas) > 0:
                            s = infeas[cdist(sol.get("X")[None, :], self.archive[infeas].get("X"))[0].argmin()]
                            others.append(s)
                    else:
                        if len(feas) > 0:
                            s = feas[cdist(sol.get("X")[None, :], self.archive[feas].get("X"))[0].argmin()]
                            others.append(s)

            doe = Population.merge(doe, self.archive[others])
            doe = DefaultDuplicateElimination().do(doe)

        self.doe = doe

        return doe

    def _set_optimum(self):
        sols = self.algorithm.opt
        if self.opt is not None:
            sols = Population.merge(sols, self.opt)
        self.opt = filter_optimum(sols, least_infeasible=True)
