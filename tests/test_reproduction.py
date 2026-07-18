"""Convergence-sanity checks: each algorithm reaches a sensible metric on its paper's problem class.

These are the coarsest fidelity tier and are deliberately framed as *ballpark* checks, not exact
reproduction of the papers' tables -- pysamoo's versions are simplifications (SAASBO is a MAP
approximation of the sparse GP, EHVI is Monte-Carlo, budgets are small), so they will not match
published numbers tightly. What they *do* guarantee is that every algorithm still converges to a
reasonable level on the kind of problem its paper targets, across several seeds -- catching a
regression that leaves an algorithm running but no longer optimizing (which a single-seed
beats-a-baseline test can miss).

Robust statistic per algorithm:

* the six low-variance algorithms are checked on the **median** over the seeds;
* TuRBO-1 and the SAASBO scaffold are acknowledged to be **variance-prone** across seeds, so they are
  checked on the **best** of the seeds (they *can* reach the optimum) -- asserting every seed
  converges would be testing noise, not correctness.

Bands are set generously (roughly 2-3x the worst value observed during calibration) so the tests are
robust to BLAS/platform jitter while still failing hard if an algorithm stops converging.
"""

import numpy as np
import pytest
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Ackley, Sphere
from pymoo.util.ref_dirs import get_reference_directions

from pysamoo.algorithms.csea import CSEA
from pysamoo.algorithms.ehvi import EHVI
from pysamoo.algorithms.krvea import KRVEA
from pysamoo.algorithms.moead_ego import MOEADEGO
from pysamoo.algorithms.parego import ParEGO
from pysamoo.algorithms.saasbo import SAASBO
from pysamoo.algorithms.tsemo import TSEMO
from pysamoo.algorithms.turbo import TuRBO

pytestmark = pytest.mark.slow

SEEDS = (1, 2, 3)
_REF3 = get_reference_directions("das-dennis", 3, n_partitions=12)
_DTLZ2 = get_problem("dtlz2", n_var=8, n_obj=3)
_IGD_ZDT1 = IGD(ZDT1(n_var=10).pareto_front())
_IGD_DTLZ2 = IGD(_DTLZ2.pareto_front(_REF3))


def _igd_zdt1(res):
    return float(_IGD_ZDT1(np.atleast_2d(np.asarray(res.F, dtype=float))))


def _igd_dtlz2(res):
    return float(_IGD_DTLZ2(np.atleast_2d(np.asarray(res.F, dtype=float))))


def _gap(res):
    """Best objective value reached (distance to the optimum at 0 for these problems)."""
    return float(np.atleast_1d(np.asarray(res.F, dtype=float)).ravel().min())


# key, problem factory, algorithm factory, evals, metric, reducer (median | min), band, paper
CASES = [
    ("parego", lambda: ZDT1(n_var=10), lambda: ParEGO(n_initial_doe=20), 80, _igd_zdt1, np.median, 0.90),
    ("ehvi", lambda: ZDT1(n_var=10), lambda: EHVI(n_initial_doe=20), 80, _igd_zdt1, np.median, 0.15),
    ("moead_ego", lambda: ZDT1(n_var=10), lambda: MOEADEGO(n_initial_doe=20), 80, _igd_zdt1, np.median, 0.90),
    ("tsemo", lambda: ZDT1(n_var=10), lambda: TSEMO(n_initial_doe=20), 80, _igd_zdt1, np.median, 1.00),
    (
        "krvea",
        lambda: _DTLZ2,
        lambda: KRVEA(ref_dirs=_REF3, n_initial_doe=40, n_infills=5),
        100,
        _igd_dtlz2,
        np.median,
        0.45,
    ),
    ("csea", lambda: _DTLZ2, lambda: CSEA(n_initial_doe=40, n_infills=5), 100, _igd_dtlz2, np.median, 0.55),
    # variance-prone -> best-of-seeds (they can reach the optimum; asserting every seed would test noise)
    ("turbo", lambda: Ackley(n_var=10), lambda: TuRBO(n_initial_doe=20), 200, _gap, np.min, 4.00),
    ("saasbo", lambda: Sphere(n_var=8), lambda: SAASBO(n_initial_doe=12), 40, _gap, np.min, 0.60),
]


@pytest.mark.parametrize(
    ("problem_fn", "algo_fn", "evals", "metric", "reduce", "band"),
    [pytest.param(*c[1:], id=c[0]) for c in CASES],
)
def test_algorithm_converges_on_paper_problem(problem_fn, algo_fn, evals, metric, reduce, band):
    """The algorithm's metric over several seeds lands in the ballpark expected for its paper problem."""
    scores = np.array(
        [metric(minimize(problem_fn(), algo_fn(), ("n_evals", evals), seed=s, verbose=False)) for s in SEEDS]
    )
    stat = float(reduce(scores))
    assert stat < band, f"{reduce.__name__}={stat:.4f} exceeds band {band} (seeds={np.round(scores, 4).tolist()})"
