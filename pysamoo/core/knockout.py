from collections import Counter

import numpy as np
from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.core.population import Population
from pymoo.core.replacement import ReplacementSurvival
from pymoo.util.dominator import get_relation


def is_better(a, b):
    assert len(a) == len(b)

    ret = np.full(len(a), False)

    a_f, a_cv, a_feas = a.get("f", "cv", "feas")
    b_f, b_cv, b_feas = b.get("f", "cv", "feas")

    # 1) Both infeasible and constraints have been improved
    ret[(~a_feas & ~b_feas) & (b_cv < a_cv)] = True

    # 2) A solution became feasible
    ret[~a_feas & b_feas] = True

    # 3) Both feasible but objective space value has improved
    ret[(a_feas & b_feas) & (b_f < a_f)] = True

    return ~ret


def comp(a, b, error=None):
    if error is not None:
        a = noisy(a, error)
        b = noisy(b, error)
    ret = is_better(a, b)
    return ret


def pcomp(sols, pairs, error=None):
    a, b = pairs.T
    a_is_better_than_b = comp(sols[a], sols[b], error=error)

    ret = np.copy(b)
    ret[a_is_better_than_b] = a[a_is_better_than_b]

    return ret


def knockout(sols, n_winners=1, error=None):
    # create a copy of all solutions to be considered
    pool = list(np.random.permutation(len(sols)))

    # until we have found a clear winner of the tournament
    while len(pool) > n_winners:

        # the list of winners in this round
        winners = []

        # make sure the pool is an even number, otherwise add one randomly (not equal to the other competing index)
        if len(pool) % 2 != 0:
            winners.append(pool[-1])
            pool = pool[:-1]

        # create the pairs that compete with each other
        pairs = np.reshape(pool, (-1, 2))

        # now add all the winners as well from this tournament
        W = pcomp(sols, pairs, error=error)
        winners.extend(W)

        # that means we have now less than we want - fill up with random solutions from pool
        if len(winners) < n_winners:
            S = set(winners)

            for k in np.random.permutation(len(sols)):

                # if not added yet then add it
                if k not in S:
                    winners.append(k)

                # we have filled up to the required number of winners -> done
                if len(winners) == n_winners:
                    break

        pool = winners

    return sols[pool]


class NoisyReplacement(ReplacementSurvival):

    def __init__(self, error, **kwargs):
        super().__init__(**kwargs)
        self.error = error

    def _do(self, problem, pop, off, **kwargs):
        return comp(off, pop, error=self.error)


def calc_prob_relation(a, b, error=None, n_comparisons=1):
    ret = []

    for k in range(n_comparisons):
        if error is not None:
            _a, _b = noisy(Population.create(a, b), error)
        else:
            _a, _b = a, b

        _rel = get_relation(_a, _b)
        ret.append(_rel)

    rel = np.random.choice([val for val, freq in Counter(ret).most_common()])

    return rel


def noisy(sols, error):
    out = {}
    for type in ["X", "F", "G", "H"]:
        out[type] = np.copy(sols.get(type))

    for (type, k), std in error.items():
        out[type][:, k] += np.random.normal(loc=0.0, scale=std, size=len(sols))

    noisy = Population.new(**out)
    TotalConstraintViolation().do(noisy, inplace=True)
    return noisy
