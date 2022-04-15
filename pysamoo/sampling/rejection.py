import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.misc import cdist
from pymoo.util.normalization import normalize


def select_points_with_maximum_distance(X, n_select, selected=[]):
    n_points, n_dim = X.shape

    # calculate the distance matrix
    D = cdist(X, X)

    # if no selection provided pick randomly in the beginning
    if len(selected) == 0:
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


class CustomLHS(LHS):

    def __init__(self, iterations=100, others=None, **kwargs) -> None:
        super().__init__(iterations=iterations, **kwargs)
        self.others = others
        self.norm_others = None

    def _do(self, problem, n_samples, **kwargs):

        if self.others is not None:
            xl, xu = problem.bounds()
            self.norm_others = normalize(self.others, xl, xu)

        return super()._do(problem, n_samples, **kwargs)

    def _calc_score(self, X):
        val = super()._calc_score(X)

        if self.norm_others is not None and len(self.norm_others) > 0:
            D = cdist(X, self.norm_others)
            val = min(val, np.min(D))

        return val


class RejectionConstrainedSampling(Sampling):

    def __init__(self,
                 func_eval_constr,
                 batch_size=None,
                 n_multiplier=2,
                 max_iter=100
                 ):
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

                sampling = CustomLHS(others=ret)

                X = sampling.do(problem, n_points).get("X")

                CV = self.func_eval_constr(X)
                is_feasible = (CV <= 0)
                X = X[is_feasible]

                ret = np.row_stack([ret, X])

        if len(ret) > n_samples:
            I = select_points_with_maximum_distance(ret, n_samples)
            ret = ret[I]

        return ret