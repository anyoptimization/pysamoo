from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from pymoo.util.roulette import RouletteWheelSelection
from sklearn.cluster import KMeans

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm


class SSANSGA2(SurrogateAssistedAlgorithm):

    def __init__(self,
                 n_infills=10,
                 surr_pop_size=100,
                 surr_n_gen=30,
                 surr_eps_elim=1e-6,
                 surr_sampling="current",
                 display=MultiObjectiveDisplay(),
                 **kwargs):

        super().__init__(display=display, **kwargs)
        self.n_infills = n_infills
        self.surr_n_gen = surr_n_gen
        self.surr_pop_size = surr_pop_size
        self.surr_eps_elim = surr_eps_elim
        self.surr_sampling = surr_sampling

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.surrogate.validate(infills)

    def _infill(self):

        self.surrogate.fit(self.archive)

        problem = self.surrogate.problem()

        if self.surr_sampling == "current":
            sampling = self.archive
        elif self.surr_sampling == "random":
            sampling = None
        else:
            raise Exception("Unknown surrogate sampling strategy.")

        algorithm = NSGA2(pop_size=self.surr_pop_size,
                          sampling=sampling
                          )

        res = minimize(problem,
                       algorithm,
                       ('n_gen', self.surr_n_gen),
                       seed=1,
                       verbose=False)

        cand = DefaultDuplicateElimination(epsilon=self.surr_eps_elim).do(res.pop, self.archive)

        if len(cand) <= self.n_infills:
            infills = Population.new(X=cand.get("X"))

        else:

            ideal = res.opt.get("F").min(axis=0)
            nadir = res.opt.get("F").max(axis=0) + 1e-16
            vals = normalize(cand.get("F"), ideal, nadir)

            kmeans = KMeans(n_clusters=self.n_infills, random_state=0).fit(vals)
            groups = [[] for _ in range(self.n_infills)]
            for k, i in enumerate(kmeans.labels_):
                groups[i].append(k)

            S = []

            for group in groups:
                if len(group) > 0:
                    fitness = cand[group].get("crowding").argsort()
                    selection = RouletteWheelSelection(fitness, larger_is_better=False)
                    I = group[selection.next()]
                    S.append(I)

            infills = Population.new(X=cand[S].get("X"))

        return infills

    def _advance(self, infills=None, **kwargs):
        self.surrogate.validate(self.archive, infills)
        super()._advance(infills, **kwargs)

    def _set_optimum(self):
        nds = NonDominatedSorting().do(self.archive.get("F"), only_non_dominated_front=True)
        self.opt = self.archive[nds]
