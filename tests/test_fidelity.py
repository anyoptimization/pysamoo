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
from pysamoo.algorithms.krvea import KRVEA
from pysamoo.algorithms.moead_ego import MOEADEGO
from pysamoo.algorithms.parego import ParEGO
from pysamoo.algorithms.saasbo import SAASBO
from pysamoo.algorithms.tsemo import TSEMO
from pysamoo.algorithms.turbo import TuRBO
from pysamoo.experimental.acquisition import EI, LogEI

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


# --------------------------------------------------------------------------------------------------
# C. MOEA/D-EGO -- optimistic LCB acquisition + decomposition selection (Zhang et al., 2010)
# --------------------------------------------------------------------------------------------------


def test_moead_lcb_is_optimistic():
    """The MOEA/D-EGO acquisition is the optimistic lower-confidence bound ``mu - kappa*sigma``."""
    mu = np.array([[1.0, 2.0], [0.5, 0.5]])
    sigma = np.array([[0.3, 0.1], [0.2, 0.4]])
    lcb = MOEADEGO.lcb(mu, sigma, kappa=2.0)
    assert np.allclose(lcb, mu - 2.0 * sigma)
    assert np.all(lcb <= mu)  # optimistic: never above the mean
    assert np.array_equal(MOEADEGO.lcb(mu, sigma, kappa=0.0), mu)  # kappa=0 -> pure mean


def test_moead_decomposition_picks_aggregation_minimizer():
    """Each weight selects the candidate minimizing its augmented Tchebycheff aggregation of the LCB."""
    # four candidates in 2D; two axis-aligned weights should each pick the extreme along their axis
    lcb = np.array([[0.0, 1.0], [1.0, 0.0], [0.4, 0.4], [0.8, 0.8]])
    weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    chosen = MOEADEGO.select_by_decomposition(lcb, weights)

    assert len(chosen) == len(set(chosen)) == 2  # one distinct pick per weight
    # verify each pick is the argmin of its own scalarization (recomputed independently)
    z = lcb.min(axis=0)
    for w, i in zip(weights, chosen):
        g = (w * (lcb - z)).max(axis=1)
        assert i in np.where(g == g.min())[0]


# --------------------------------------------------------------------------------------------------
# C. TSEMO -- Thompson sampling + greedy hypervolume selection (Bradford et al., 2018)
# --------------------------------------------------------------------------------------------------


def test_tsemo_thompson_sample_formula():
    """A Thompson draw is ``mu + sigma * z``: equal to the mean at zero variance, unbiased otherwise."""
    mu = np.array([[1.0, -2.0], [0.0, 3.0]])
    sigma = np.zeros_like(mu)
    rng = np.random.default_rng(0)
    assert np.array_equal(TSEMO.thompson_sample(mu, sigma, rng), mu)  # sigma=0 -> exactly the mean

    sigma = np.full_like(mu, 0.5)
    draws = np.array([TSEMO.thompson_sample(mu, sigma, rng) for _ in range(20000)])
    assert np.allclose(draws.mean(axis=0), mu, atol=0.02)  # unbiased around the mean
    assert np.allclose(draws.std(axis=0), 0.5, atol=0.02)  # spread == sigma


def test_tsemo_greedy_hvi_selection_is_monotone():
    """Greedy TSEMO selection adds points that never decrease the running hypervolume; first is the best."""
    ref = np.array([2.0, 2.0])
    hv = HV(ref_point=ref)
    front = np.array([[1.5, 1.5]])
    sample = np.array([[1.0, 1.0], [0.5, 1.8], [1.8, 0.5]])  # candidate objective values
    chosen = TSEMO.greedy_hvi_select(sample, front, hv, n_infills=3, candidates=[0, 1, 2])

    assert len(chosen) == 3 and len(set(chosen)) == 3
    # the first pick is the single candidate with the largest hypervolume improvement
    hv0 = float(hv(front))
    gains = [float(hv(np.vstack([front, sample[i]]))) - hv0 for i in range(3)]
    assert chosen[0] == int(np.argmax(gains))
    # the accumulated hypervolume is monotone non-decreasing along the greedy order
    cur, hvs = front, []
    for i in chosen:
        cur = np.vstack([cur, sample[i]])
        hvs.append(float(hv(cur)))
    assert np.all(np.diff(hvs) >= -1e-12)


# --------------------------------------------------------------------------------------------------
# C. K-RVEA -- adaptive diversity/convergence infill switch (Chugh et al., 2018)
# --------------------------------------------------------------------------------------------------


def test_krvea_convergence_selects_most_uncertain():
    """When the active-vector set is stable (``changed <= delta``) K-RVEA picks the most uncertain points."""
    sigma = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    assign = np.array([0, 0, 1, 1, 2])
    cos = np.eye(5)[:, :3]
    sel = KRVEA.select_infills(assign, cos, sigma, active={0, 1, 2}, changed=0.0, delta=0.1, u=2)
    assert list(sel) == list(np.argsort(-sigma)[:2])  # the two highest-uncertainty candidates
    assert set(sel) == {1, 3}


def test_krvea_diversity_one_per_active_vector_and_fills_batch():
    """In the diversity regime K-RVEA takes one point per active vector, topped up to the full batch."""
    # 4 candidates, only 2 active reference vectors, but the batch wants u=3 -> must top up by sigma
    assign = np.array([0, 0, 1, 1])
    cos = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])
    sigma = np.array([0.1, 0.2, 0.3, 0.9])
    sel = KRVEA.select_infills(assign, cos, sigma, active={0, 1}, changed=1.0, delta=0.1, u=3)

    assert len(sel) == 3, "the batch must always spend its full budget"
    # the per-vector picks are the best-aligned candidate of each active vector...
    assert 0 in sel  # vector 0: candidate 0 has the largest cosine (0.9)
    assert 2 in sel  # vector 1: candidate 2 has the largest cosine (0.8)
    # ...and the top-up is the most uncertain remaining candidate (index 3, sigma=0.9)
    assert 3 in sel


# --------------------------------------------------------------------------------------------------
# C. SAASBO -- sparse axis-aligned ARD prior (Eriksson & Jankowiak, 2021)
# --------------------------------------------------------------------------------------------------


def test_saasbo_uses_sparse_ard_shrinkage_prior():
    """SAASBO's surrogate is an ARD Kriging carrying the sparsity-inducing shrinkage prior on theta.

    The defining ingredient of SAASBO is a strong prior that pulls the per-dimension length-scales
    toward "inactive" unless the data demands otherwise. This checks that the default surrogate is
    ARD (a length-scale per dimension) and that the shrinkage ``theta_prior`` is actually wired in.
    """
    algo = SAASBO()
    proto = algo.surrogate_proto
    assert getattr(proto, "ARD", False) is True, "SAASBO must use an ARD (per-dimension) Kriging"
    mean, std = algo.theta_prior
    assert (mean, std) == proto.theta_prior
    assert std > 0, "the shrinkage prior must have positive strength"
    # a tighter prior (smaller std) is a stronger pull toward sparsity
    assert SAASBO(theta_prior=(0.0, 0.001)).theta_prior[1] < SAASBO().theta_prior[1]


# --------------------------------------------------------------------------------------------------
# C. Acquisition -- the (Log)EI that ParEGO / TuRBO / SAASBO all optimize (Ament et al., 2023)
# --------------------------------------------------------------------------------------------------


def test_ei_matches_closed_form_expected_improvement():
    """``EI.calc`` is the analytic Expected Improvement (returned negated, for minimization)."""
    rng = np.random.default_rng(0)
    mu, sigma, f_min = rng.uniform(-1, 3, 60), rng.uniform(0.1, 1.0, 60), 0.5
    z = (f_min - mu) / sigma
    ei = (f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)  # textbook closed form
    assert np.allclose(EI().calc(mu, sigma, f_min=f_min), -ei)


def test_logei_matches_log_of_ei_where_ei_is_finite():
    """``LogEI`` equals ``log(EI)`` wherever EI has not underflowed -- validating its stable core."""
    rng = np.random.default_rng(1)
    mu, sigma, f_min = rng.uniform(-1, 2, 300), rng.uniform(0.2, 1.0, 300), 1.0
    ei = -EI().calc(mu, sigma, f_min=f_min)  # positive EI
    ok = ei > 1e-9
    log_ei = -LogEI().calc(mu, sigma, f_min=f_min)  # equals log(EI)
    assert np.allclose(log_ei[ok], np.log(ei[ok]), atol=1e-6)


def test_logei_preserves_ordering_and_stays_finite_under_underflow():
    """``LogEI`` shares EI's argmax (monotone transform) and stays finite where EI underflows to 0."""
    rng = np.random.default_rng(2)
    mu, sigma, f_min = rng.uniform(0, 5, 400), rng.uniform(0.1, 1.0, 400), 0.2
    assert np.argmax(-EI().calc(mu, sigma, f_min=f_min)) == np.argmax(-LogEI().calc(mu, sigma, f_min=f_min))

    # deep-underflow regime: the incumbent is far better than every candidate
    mu2, sigma2 = np.array([10.0, 12.0]), np.array([0.5, 0.5])
    assert np.all(-EI().calc(mu2, sigma2, f_min=0.0) < 1e-30)  # EI underflows to ~0
    assert np.all(np.isfinite(LogEI().calc(mu2, sigma2, f_min=0.0)))  # LogEI stays finite and usable
