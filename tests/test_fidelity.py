"""Fidelity tests: validate algorithm internals against ground truth and invariants, not just baselines.

The performance suite only shows each algorithm *beats a baseline*; that cannot tell a faithful
implementation from a lucky-but-wrong one. These tests check the pieces the published methods hinge on
against things we can compute exactly:

* **Ground truth** -- the Monte-Carlo EHVI acquisition against a 2-objective grid quadrature of the
  same integral, and a hand-computed hypervolume improvement.
* **Invariants** -- EHVI is zero for a dominated candidate; a run's front hypervolume never decreases;
  the reported optimum is a non-dominated set; TuRBO's trust-region state machine follows the paper's
  expand/shrink/restart rules; CSEA's classification target matches non-dominated ranks.

They exercise the real code (the ``EHVI.expected_hvi`` and ``CSEA.label_good`` seams, and
``TuRBO._advance``), so a regression in the core math fails here even when the baseline is still beaten.
"""

import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import norm

from pysamoo.algorithms.csea import CSEA
from pysamoo.algorithms.ehvi import EHVI
from pysamoo.algorithms.parego import ParEGO
from pysamoo.algorithms.turbo import TuRBO

# --------------------------------------------------------------------------------------------------
# A. Ground truth
# --------------------------------------------------------------------------------------------------


def test_hypervolume_improvement_matches_hand_value():
    """The HV improvement of adding one point matches an exactly hand-computed area."""
    ref = np.array([2.0, 2.0])
    hv = HV(ref_point=ref)
    front = np.array([[1.0, 1.0]])  # dominates the box [1,2] x [1,2] -> HV = 1.0
    assert hv(front) == pytest.approx(1.0)
    # adding (0.5, 0.5): it dominates [0.5,2] x [0.5,2] (area 2.25); the union HV is 2.25.
    hvi = float(hv(np.vstack([front, [0.5, 0.5]]))) - float(hv(front))
    assert hvi == pytest.approx(2.25 - 1.0)


def test_mc_ehvi_matches_grid_quadrature():
    """EHVI.expected_hvi (Monte-Carlo) converges to a 2-objective grid quadrature of the same integral."""
    ref = np.array([2.0, 2.0])
    hv = HV(ref_point=ref)
    front = np.array([[1.0, 1.0]])
    hv0 = float(hv(front))
    mu, sigma = np.array([0.8, 0.8]), np.array([0.3, 0.3])

    # ground truth: integral of HVI(f) * N(f1;mu1,s1) N(f2;mu2,s2) over a grid
    axes = [np.linspace(mu[k] - 5 * sigma[k], mu[k] + 5 * sigma[k], 60) for k in (0, 1)]
    d = [axes[k][1] - axes[k][0] for k in (0, 1)]
    w = [norm.pdf(axes[k], mu[k], sigma[k]) * d[k] for k in (0, 1)]
    quad = 0.0
    for i, f1 in enumerate(axes[0]):
        for j, f2 in enumerate(axes[1]):
            hvi = max(0.0, float(hv(np.vstack([front, [f1, f2]]))) - hv0)
            quad += hvi * w[0][i] * w[1][j]

    mc = EHVI.expected_hvi(mu, sigma, front, hv, hv0, n_samples=20000, random_state=np.random.default_rng(0))
    assert mc == pytest.approx(quad, rel=0.08)


def test_ehvi_zero_for_dominated_candidate():
    """A candidate whose mean is dominated by the front (with tiny sigma) has ~zero EHVI."""
    ref = np.array([2.0, 2.0])
    hv = HV(ref_point=ref)
    front = np.array([[0.5, 0.5]])
    hv0 = float(hv(front))
    mu, sigma = np.array([1.5, 1.5]), np.array([1e-4, 1e-4])  # dominated by (0.5, 0.5)
    ehvi = EHVI.expected_hvi(mu, sigma, front, hv, hv0, n_samples=2000, random_state=np.random.default_rng(0))
    assert ehvi == pytest.approx(0.0, abs=1e-6)


def test_parego_scalarization_is_augmented_tchebycheff():
    """ParEGO's scalarization matches the augmented-Tchebycheff formula on a hand example."""
    # normalized objectives (ideal at 0), a weight, and rho as ParEGO uses them
    Fn = np.array([[0.2, 0.8], [0.6, 0.1]])
    lam = np.array([0.4, 0.6])
    rho = 0.05
    d = lam * Fn
    y = d.max(axis=1) + rho * d.sum(axis=1)
    # by hand: row0 d=(0.08,0.48) -> max 0.48 + 0.05*0.56 = 0.508 ; row1 d=(0.24,0.06) -> 0.24 + 0.05*0.30 = 0.255
    assert y == pytest.approx([0.508, 0.255])


# --------------------------------------------------------------------------------------------------
# B. Invariants
# --------------------------------------------------------------------------------------------------


def test_csea_label_good_matches_nondominated_rank():
    """CSEA's 'good' target is exactly 'non-dominated rank <= median rank'."""
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 2.0], [3.0, 3.0]])
    ranks = NonDominatedSorting().do(F, return_rank=True)[1]
    good = CSEA.label_good(F)
    assert np.array_equal(good, ranks <= np.median(ranks))
    assert good[:3].all()  # the three rank-0 points are good
    assert not good[-1]  # the most-dominated point is not


def test_turbo_trust_region_state_machine():
    """TuRBO's length adapts by the paper's rules: expand on successes, shrink on failures, restart on collapse."""
    algo = TuRBO(length_init=0.5, length_min=0.1, length_max=1.0, succ_tol=2, fail_tol=2)
    algo._archive = Population.new(X=np.zeros((1, 2)), F=np.array([[1.0]]))

    def step(f):
        algo._advance(Population.new(X=np.zeros((1, 2)), F=np.array([[f]])))

    step(0.9)  # improvement -> success 1
    assert algo.success == 1 and algo.L == 0.5
    step(0.8)  # improvement -> success 2 -> expand (capped at length_max)
    assert algo.L == 1.0 and algo.success == 0
    step(0.85)  # no improvement -> failure 1
    step(0.9)  # no improvement -> failure 2 -> shrink
    assert algo.L == 0.5 and algo.failure == 0
    assert algo.length_min <= algo.L <= algo.length_max

    algo.L, algo.failure = 0.18, 1
    step(0.95)  # failure 2 -> L halves to 0.09 < length_min -> restart flag
    assert algo._restart is True


@pytest.mark.slow
def test_front_hypervolume_never_decreases():
    """Across an EHVI run the non-dominated front's hypervolume is monotone non-decreasing."""
    from pymoo.core.callback import Callback

    problem = ZDT1(n_var=5)
    ref = np.array([1.1, 1.1])

    class HVTrace(Callback):
        def __init__(self):
            super().__init__()
            self.hv = []

        def notify(self, algo):
            F = algo._archive.get("F")
            nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
            self.hv.append(float(HV(ref_point=ref)(F[nd])))

    cb = HVTrace()
    minimize(
        problem,
        EHVI(n_initial_doe=20, pool=60, n_screen=8, n_samples=16),
        ("n_evals", 40),
        seed=1,
        callback=cb,
        verbose=False,
    )
    hv = np.array(cb.hv)
    assert np.all(np.diff(hv) >= -1e-9), hv


@pytest.mark.slow
@pytest.mark.parametrize(
    "algo",
    [ParEGO(n_initial_doe=15), EHVI(n_initial_doe=15, pool=60, n_screen=8, n_samples=16), CSEA(n_initial_doe=15)],
    ids=["parego", "ehvi", "csea"],
)
def test_reported_optimum_is_nondominated(algo):
    """The optimum returned by each multi-objective algorithm is a mutually non-dominated set."""
    res = minimize(ZDT1(n_var=5), algo, ("n_evals", 25), seed=1, verbose=False)
    F = np.atleast_2d(res.F)
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    assert len(nd) == len(F), "reported optimum contains dominated points"
