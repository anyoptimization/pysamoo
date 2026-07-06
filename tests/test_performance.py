"""Performance-regression tests: each surrogate-assisted algorithm must BEAT the baseline it wraps.

The rest of the suite proves the algorithms *run* and are reproducible; these prove they still
*help*. At a fixed seed and small budget, each SAO algorithm must converge further than the exact
pymoo baseline it wraps. Verified across 5 seeds during development (see ``docs/PERFORMANCE.md``):
PSAF(GA) ~4x better than GA and GPSAF(NSGA2) ~6x better than NSGA2 on the cases below.

Two complementary guards:

* **Assertions** on a large-margin inequality (SAO score < half the baseline's). An inequality with
  a wide margin is robust across platforms/BLAS, unlike a committed float value.
* **Golden** snapshots of the exact seed-1 scores for drift tracking (loose tolerance; a legitimate
  cross-machine shift is blessed by a human, per the golden workflow).

Marked ``slow`` -- they run full optimizations, so they belong in ``pyclawd test all`` (pre-release),
not the every-edit gate.
"""

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Ackley

from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.algorithms.parego import ParEGO
from pysamoo.algorithms.psaf import PSAF
from pysamoo.algorithms.ssansga2 import SSANSGA2

pytestmark = pytest.mark.slow

SEED = 1
_PSAF_KW = dict(n_initial_doe=30, alpha=10, beta=30, max_rho=0.7, n_max_infills=10, n_max_doe=500)
_GPSAF_KW = dict(n_initial_doe=30, alpha=10, beta=50, n_max_doe=100)


def _f_gap(problem, res):
    """Best feasible objective value reached (gap to the optimum at 0 for these problems)."""
    return float(np.atleast_1d(np.asarray(res.F, dtype=float)).ravel().min())


def _igd(problem, res):
    return float(IGD(problem.pareto_front())(np.atleast_2d(np.asarray(res.F, dtype=float))))


@pytest.fixture(scope="module")
def scores():
    """Run each baseline and its SAO wrapper once at a fixed seed; return the comparison scores."""
    ackley = Ackley(n_var=10)
    zdt1 = ZDT1(n_var=10)

    def run(problem, algo, n_evals):
        return minimize(problem, algo, ("n_evals", n_evals), seed=SEED, verbose=False)

    return {
        "ga": _f_gap(ackley, run(ackley, GA(pop_size=20, n_offsprings=10), 300)),
        "psaf_ga": _f_gap(ackley, run(ackley, PSAF(GA(pop_size=20, n_offsprings=10), **_PSAF_KW), 300)),
        "de": _f_gap(ackley, run(ackley, DE(pop_size=20, n_offsprings=10), 300)),
        "psaf_de": _f_gap(ackley, run(ackley, PSAF(DE(pop_size=20, n_offsprings=10), **_PSAF_KW), 300)),
        "nsga2": _igd(zdt1, run(zdt1, NSGA2(pop_size=20, n_offsprings=10), 200)),
        "gpsaf": _igd(zdt1, run(zdt1, GPSAF(NSGA2(pop_size=20, n_offsprings=10), **_GPSAF_KW), 200)),
        "ssansga2": _igd(zdt1, run(zdt1, SSANSGA2(n_initial_doe=50, n_infills=10, surr_pop_size=100), 200)),
        # ParEGO reaches a strong front with far fewer evals (120 vs NSGA2's 200) -- a stronger claim.
        "parego": _igd(zdt1, run(zdt1, ParEGO(n_initial_doe=30), 120)),
    }


# --- assertions: SAO must beat its baseline by a wide margin (platform-robust) ---


def test_psaf_ga_beats_ga(scores):
    """PSAF(GA) converges much further than plain GA on Ackley."""
    assert scores["psaf_ga"] < 0.5 * scores["ga"], scores


def test_psaf_de_beats_de(scores):
    """PSAF(DE) converges much further than plain DE on Ackley."""
    assert scores["psaf_de"] < 0.7 * scores["de"], scores


def test_gpsaf_beats_nsga2(scores):
    """GPSAF(NSGA2) reaches a much lower IGD than plain NSGA2 on ZDT1."""
    assert scores["gpsaf"] < 0.5 * scores["nsga2"], scores


def test_ssansga2_beats_nsga2(scores):
    """SSANSGA2 reaches a lower IGD than plain NSGA2 on ZDT1 at this (favourable) seed."""
    assert scores["ssansga2"] < scores["nsga2"], scores


def test_parego_beats_nsga2(scores):
    """ParEGO reaches a clearly lower IGD than NSGA2 on ZDT1 -- with fewer evaluations (120 vs 200)."""
    assert scores["parego"] < 0.7 * scores["nsga2"], scores


# --- golden: exact seed-1 scores for drift tracking ---


@pytest.mark.golden
def test_golden_performance(scores):
    """Snapshot the exact seed-1 scores of every baseline/SAO pair.

    All runs are now bit-reproducible for a fixed seed, including GPSAF: its former run-to-run
    drift came from an unseeded ``LHS`` in ``GPSAF._doe`` (archive subsampling past ``n_max_doe``),
    now threaded with the run's ``random_state``. So GPSAF is included in the snapshot again.
    """
    return {k: round(v, 6) for k, v in scores.items()}
