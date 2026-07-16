"""GPSAF — the generalized probabilistic surrogate-assisted framework algorithm."""

from copy import deepcopy

import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import default_termination
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.tournament import compare
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.util.dominator import get_relation
from pymoo.util.misc import cdist, norm_eucl_dist
from pymoo.util.optimum import filter_optimum

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
from pysamoo.core.knockout import noisy

# =========================================================================================================
# Display
# =========================================================================================================


class GPSAFOutput(Output):
    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)
        self.output = output

        self.n_influenced = Column(name="n_influenced")

        self.mode = None
        self.mae = Column(name="mae")
        self.model = Column(name="model", width=60)

        self.mae_f1 = Column(name="mae f1")
        self.mae_f2 = Column(name="mae f2")

    def initialize(self, algorithm):
        self.output.initialize(algorithm)
        self.columns = [e for e in self.output.columns if e.name != "f_avg"] + [self.n_influenced]

        problem = algorithm.problem
        if problem.n_obj == 1 and problem.n_constr == 0:
            self.mode = "soo"
            self.columns += [self.mae, self.model]
        elif problem.n_obj == 2 and problem.n_constr == 0:
            self.mode = "moo"
            self.columns += [self.mae_f1, self.mae_f2]

    def update(self, algorithm):
        gpasf = algorithm
        self.output.update(gpasf)

        if gpasf.n_gen > 1:
            surr_infills = Population.create(*gpasf.infills.get("created_by"))
            n_influenced = sum(surr_infills.get("type") == "trace")
            self.n_influenced.set(f"{n_influenced}/{len(surr_infills)}")

        if self.mode == "soo":
            target = gpasf.surrogate.targets[0]
            self.mae.set(target.performance("mae"))
            self.model.set(target.best)
        elif self.mode == "moo":
            perf = gpasf.surrogate.performance("mae")
            if ("F", 0) in perf:
                self.mae_f1.set(perf[("F", 0)])
            if ("F", 1) in perf:
                self.mae_f2.set(perf[("F", 1)])


# =========================================================================================================
# Algorithm
# =========================================================================================================


class GPSAF(SurrogateAssistedAlgorithm):
    def __init__(self, algorithm, alpha=5, beta=30, rho=None, n_max_doe=100, n_max_infills=np.inf, **kwargs):

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

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)

        # set the default termination to the proto type
        if self.proto.termination is None:
            self.proto.termination = default_termination(problem)

        # setup the underlying algorithm
        kwargs["seed"] = self.seed
        self.proto.setup(problem, **kwargs)

        # copy the algorithm object to get started
        self.algorithm = deepcopy(self.proto)

        # customize the display to show the surrogate influence
        self.display = self.algorithm.display
        self.display.output = GPSAFOutput(self.algorithm.output)
        self.algorithm.display = lambda _: None

        # define the survival for the individuals to keep
        self.survival = FitnessSurvival() if problem.n_obj == 1 else RankAndCrowdingSurvival()

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)

        # validate and check different surrogates to find the best
        self.surrogate.validate(infills, random_state=self.random_state)

        # feed back the fittest individuals to the algorithm
        self.algorithm.advance(infills=infills)

    def _infill(self):

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
                if self.random_state.random() <= rho:
                    # if it should be replaced find the ONE solution from the pool
                    biased = self._infill_prob_tourn(pool, error=error, n_winners=1)[0]

                    # check the distance to existing solutions
                    closest = norm_eucl_dist(self.problem, biased.get("X"), self._archive.get("X")).min()

                    # if the solution is in fact new
                    if closest > eps:
                        # now actually set the value to the infills
                        infills[i] = biased

        # if beta is zero, then simply take the results from the alpha phase
        else:
            infills = influenced

        # now copy over the infills and set them to have never been evaluated
        ret = deepcopy(infills)
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
            influenced = self._infill_prob_tourn(influenced, error=error, n_winners=self.n_max_infills)

        # do the tournament for each alpha
        for _ in range(self.alpha - 1):
            # create a second pool and actually do the tournament
            others = self.algorithm.infill()
            if len(others) > len(influenced):
                I = self.random_state.permutation(len(others))[: len(influenced)]
                others = others[I]

            Evaluator().eval(problem, others)

            # for each offspring see if we do the surrogate tournament
            for i in range(len(influenced)):
                # if the competitor is not worse it will take the lead
                if get_relation(influenced[i], others[i]) < 1:
                    influenced[i] = others[i]

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

        # disable the termination to have enough iterations to continue
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

    def _infill_prob_tourn(self, sols, n_winners=1, error=None):
        # a knockout tournament with probabilistic comparisons under surrogate-prediction noise

        # create a copy of all solutions to be considered
        pool = list(range(len(sols)))

        # until we have found a clear winner of the tournament
        while True:
            # always shuffle the pool to have random tournaments
            self.random_state.shuffle(pool)

            # make sure the pool is an even number
            if len(pool) % 2 != 0:
                pool.append(self.random_state.choice(pool))

            # create the pairs that compete with each other
            pairs = np.reshape(np.array(pool), (-1, 2))

            # prepare the next pool already containing all the winners
            winners = []

            # create a solution pool with noise
            sols_with_noise = noisy(sols, error, random_state=self.random_state)

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
                        k = compare(
                            i,
                            a.get("k"),
                            j,
                            b.get("k"),
                            method="larger_is_better",
                            return_random_if_equal=True,
                            random_state=self.random_state,
                        )
                    else:
                        k = self.random_state.choice([i, j])

                    winners.append(k)

            if len(winners) <= n_winners:
                if len(winners) < n_winners:
                    H = set(winners)

                    self.random_state.shuffle(pool)

                    for k in pool:
                        if k not in H:
                            winners.append(k)
                            H.add(k)

                        if len(winners) == n_winners:
                            break

                return sols[winners]

            pool = winners

    def _advance(self, infills=None, **kwargs):

        # re-select the surrogate model (lazily; see nth_validate)
        self.revalidate(trn=self.doe, tst=infills)

        # make a step in the main algorithm with high-fidelity solutions
        self.algorithm.advance(infills=infills, **kwargs)

        super()._advance(infills=infills, **kwargs)

    def _doe(self):
        n_max_doe = self.n_max_doe

        doe = self._archive

        if len(doe) > n_max_doe:
            # Thread the run's Generator into the LHS: without it, pymoo falls back to an unseeded
            # default_rng(), so the cluster centers -- and therefore which archive points are kept
            # for the surrogate once the archive exceeds n_max_doe -- were random each run. That was
            # the sole source of GPSAF's run-to-run irreproducibility (it only bit past n_max_doe).
            center = LHS().do(self.problem, n_max_doe, random_state=self.random_state).get("X")
            A = cdist(doe.get("X"), center).argmin(axis=1)

            cluster = [[] for _ in range(n_max_doe)]
            for i, a in enumerate(A):
                cluster[a].append(i)

            doe = []
            for c in cluster:
                if len(c) > 0:
                    s = self.random_state.choice(c)
                    doe.append(s)

            doe = self._archive[doe]
            doe = Population.merge(doe, self.algorithm.pop)
            doe = Population.merge(doe, self.algorithm.opt)
            doe = Population.merge(doe, self.opt)
            doe = DefaultDuplicateElimination().do(doe)

            # if we have constraints also select the closest infeas. or feas. solution
            others = []
            if self.problem.has_constraints():
                is_feas = self._archive.get("feas")
                feas, infeas = np.where(is_feas)[0], np.where(~is_feas)[0]

                for sol in doe:
                    if sol.get("feas"):
                        if len(infeas) > 0:
                            s = infeas[cdist(sol.get("X")[None, :], self._archive[infeas].get("X"))[0].argmin()]
                            others.append(s)
                    else:
                        if len(feas) > 0:
                            s = feas[cdist(sol.get("X")[None, :], self._archive[feas].get("X"))[0].argmin()]
                            others.append(s)

            doe = Population.merge(doe, self._archive[others])
            doe = DefaultDuplicateElimination().do(doe)

        self.doe = doe

        return doe

    def _set_optimum(self):
        sols = self.algorithm.opt
        if self.opt is not None:
            sols = Population.merge(sols, self.opt)
        self.opt = filter_optimum(sols, least_infeasible=True)
