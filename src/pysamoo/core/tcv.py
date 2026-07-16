"""Total constraint-violation aggregation utilities."""

from collections.abc import Callable

import numpy as np
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.misc import at_least_2d_array


class TotalConstraintViolation:
    def __init__(
        self,
        ieq_eps: float = 0.0,
        ieq_pow: float | None = None,
        ieq_scale: np.ndarray | None = None,
        eq_eps: float = 1e-4,
        eq_pow: float | None = None,
        eq_scale: np.ndarray | None = None,
        aggr_func: Callable = np.mean,
        feas_eps: float = 0.0,
    ):
        """Aggregate inequality/equality constraints into a single total-constraint-violation value.

        Args:
            ieq_eps: Allowed violation of an inequality constraint (usually 0, may be relaxed during a run).
            ieq_pow: Power each inequality-constraint violation is raised to.
            ieq_scale: Per-constraint scaling; the violations are divided by it (useful for differently
                scaled constraints that would otherwise bias the aggregation).
            eq_eps: Permitted violation of an equality constraint (a small epsilon).
            eq_pow: Power each equality-constraint violation is raised to.
            eq_scale: Same as ``ieq_scale`` but for equality constraints.
            aggr_func: Aggregation applied across constraints to obtain the total violation.
            feas_eps: The violation threshold below which a solution counts as feasible.
        """

        super().__init__()

        self.ieq_beta = ieq_pow
        self.ieq_eps = ieq_eps
        self.ieq_scale = ieq_scale

        self.eq_beta = eq_pow
        self.eq_eps = eq_eps
        self.eq_scale = eq_scale

        self.aggr_func = aggr_func

        self.feas_eps = feas_eps

    def calc(self, G: np.ndarray = None, H: np.ndarray = None, return_feas=False):

        # convert all constraints to one big array
        C = []

        if G is not None:
            G = at_least_2d_array(G, extend_as="r")
            cv_ieq = g_to_cv(G, self.ieq_eps, beta=self.ieq_beta, scale=self.ieq_scale)
            C.append(cv_ieq)

        if H is not None:
            H = at_least_2d_array(H, extend_as="r")
            cv_eq = g_to_cv(np.abs(H), self.eq_eps, beta=self.eq_beta, scale=self.eq_scale)
            # cv_eq = g_to_cv(H ** 2, self.eq_eps ** 2, beta=self.eq_pow, scale=self.eq_scale)
            C.append(cv_eq)

        # simply return None if there are no constraints
        if len(C) == 0:
            return None

        stacked = np.column_stack(C)

        # calculate the total constraint violation
        tcv = self.aggr_func(stacked, axis=1)

        if return_feas:
            return tcv, tcv <= self.feas_eps
        else:
            return tcv

    def do(self, pop, inplace=True):

        # this way the total constraint violation calculation also works for an individual
        if isinstance(pop, Individual):
            pop = Population.create(pop)

        # do the actual calculations to get the total constraint violations
        G, H = pop.get("G", "H")

        if G.shape[1] == 0 and H.shape[1] == 0:
            tcv = np.zeros(len(pop))
        else:
            tcv = self.calc(G, H)

        # set the cv values inplace directly
        if inplace:
            pop.set("CV", tcv[:, None])
            pop.set("feas_eps", self.feas_eps)

        return tcv


def g_to_cv(g, eps, beta=None, scale=None):
    # subtract eps to allow some violation and then zero out all values less than zero
    g = np.maximum(0.0, g - eps)

    # apply scaling if necessary
    if scale is not None:
        # allow a scalar value as input
        if not isinstance(scale, np.ndarray):
            scale = np.full(g.shape[1], scale)

        # make sure to only use positive scaling and not zero
        I = np.where(scale > 0)[0]
        g[:, I] = g[:, I] / scale[I]

    # if a pow factor has been provided
    if beta is not None:
        g = g**beta

    return g
