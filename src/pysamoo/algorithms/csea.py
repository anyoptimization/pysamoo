"""CSEA -- Classification-based Surrogate-assisted Evolutionary Algorithm for expensive many-objective problems."""

import numpy as np
from pymoo.core.population import Population
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.neighbors import KNeighborsClassifier

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm, default_n_doe


class CSEA(SurrogateAssistedAlgorithm):
    """CSEA (Pan et al., 2019): a *classification* surrogate that screens offspring for evaluation.

    Instead of regressing each objective, CSEA trains a single classifier to answer a cheaper
    question -- "is this design one of the good ones?" -- which sidesteps the accuracy demands of
    regression and scales naturally to many objectives. Each iteration labels the archive (good =
    better-than-median non-dominated rank), fits a K-nearest-neighbour classifier on the decision
    vectors, generates a large pool of genetic offspring (recombination + mutation of the archive),
    and evaluates the ``n_infills`` offspring the classifier judges most likely to be good. Reuses
    scikit-learn's classifier and pymoo's non-dominated sorting.

    Args:
        n_infills: True-function evaluations selected per iteration.
        n_offspring: Genetic offspring generated and classified per iteration.
        n_neighbors: Neighbours for the K-nearest-neighbour classifier.
        p_mut: Per-coordinate mutation probability (defaults to ``1/n_var``).
    """

    def __init__(self, n_infills=5, n_offspring=200, n_neighbors=5, p_mut=None, output=None, **kwargs):
        super().__init__(output=output if output is not None else MultiObjectiveOutput(), **kwargs)
        self.n_infills = n_infills
        self.n_offspring = n_offspring
        self.n_neighbors = n_neighbors
        self.p_mut = p_mut

    def _setup(self, problem, **kwargs):
        if self.n_initial_doe is None:
            self.n_initial_doe = min(self.n_initial_max_doe, default_n_doe(problem.n_var))

    def _offspring(self, X, xl, xu, rng):
        # genetic offspring: blend recombination of two random parents + Gaussian mutation
        n, d = self.n_offspring, X.shape[1]
        p1, p2 = X[rng.integers(len(X), size=n)], X[rng.integers(len(X), size=n)]
        beta = rng.random((n, d))
        child = beta * p1 + (1.0 - beta) * p2
        p_mut = self.p_mut if self.p_mut is not None else 1.0 / d
        mut = rng.random((n, d)) < p_mut
        child = child + mut * (0.1 * (xu - xl) * rng.standard_normal((n, d)))
        return np.clip(child, xl, xu)

    def _infill(self):
        X, F = self._archive.get("X", "F")
        problem = self.problem
        xl, xu = problem.xl, problem.xu
        span = np.maximum(xu - xl, 1e-12)
        rng = self.random_state

        cand = self._offspring(X, xl, xu, rng)

        # label the archive: "good" = non-dominated rank at or below the median rank
        ranks = NonDominatedSorting().do(F, return_rank=True)[1]
        good = ranks <= np.median(ranks)

        # if the labels are degenerate (all one class), no classifier is possible -> pick at random
        if good.all() or (~good).all():
            sel = rng.permutation(len(cand))[: self.n_infills]
            return Population.new(X=cand[sel])

        # KNN classifier on normalized decision vectors; score offspring by P(good)
        clf = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, int(good.sum()), int((~good).sum())) or 1)
        clf.fit((X - xl) / span, good.astype(int))
        prob = clf.predict_proba((cand - xl) / span)
        good_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
        score = prob[:, good_col]

        sel = np.argsort(-score)[: self.n_infills]
        return Population.new(X=cand[sel])

    def _set_optimum(self):
        nds = NonDominatedSorting().do(self._archive.get("F"), only_non_dominated_front=True)
        self.opt = self._archive[nds]
