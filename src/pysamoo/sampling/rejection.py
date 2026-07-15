"""Rejection-based constrained sampling with maximum-distance selection."""

import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.misc import cdist


def select_points_with_maximum_distance(X, n_select, selected=None):
    n_points, n_dim = X.shape

    # calculate the distance matrix
    D = cdist(X, X)

    # if no selection provided pick randomly in the beginning
    if not selected:
        selected = [np.random.randint(len(X))]

    # create variables to store what selected and what not
    not_selected = [i for i in range(n_points) if i not in selected]

    # remove unnecessary points
    dist_to_closest_selected = D[:, selected].min(axis=1)

    # now select the points until sufficient ones are found
    while len(selected) < n_select:
        # find point that has the maximum distance to all others
        index_in_not_selected = dist_to_closest_selected[not_selected].argmax()
        I = not_selected[index_in_not_selected]

        # add the closest distance to selected point
        is_closer = D[I] < dist_to_closest_selected
        dist_to_closest_selected[is_closer] = D[I][is_closer]

        # add it to the selected and remove from not selected
        selected.append(int(I))
        not_selected = np.delete(not_selected, index_in_not_selected)

    return selected


class RejectionConstrainedSampling(Sampling):
    def __init__(self, func_eval_constr, batch_size=None, n_multiplier=2, max_iter=100):
        super().__init__()
        self.max_iter = max_iter
        self.n_multiplier = n_multiplier
        self.batch_size = batch_size
        self.func_eval_constr = func_eval_constr

    def _do(self, problem, n_samples, **kwargs):

        n_points = self.batch_size
        if n_points is None:
            n_points = 2 * n_samples

        ret = np.zeros((0, problem.n_var))

        for k in range(self.max_iter):
            if len(ret) >= self.n_multiplier * n_samples:
                break

            else:
                sampling = LHS(iterations=100)

                X = sampling.do(problem, n_points).get("X")

                CV = self.func_eval_constr(X)
                is_feasible = CV <= 0
                X = X[is_feasible]

                ret = np.vstack([ret, X])

        if len(ret) > n_samples:
            I = select_points_with_maximum_distance(ret, n_samples)
            ret = ret[I]

        return ret
