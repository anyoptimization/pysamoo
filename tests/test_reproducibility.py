"""Reproducibility tests: the same seed must give the same result.

These guard the random_state threading. Surrogate-assisted runs were previously
non-reproducible because selection/infill code used the global numpy/`random`
state (which pymoo 0.6.1 no longer seeds). Each algorithm now threads the run's
``self.random_state`` Generator through every stochastic site, so a fixed seed
yields bit-identical results.

We assert *equality across two runs* rather than against a committed baseline:
that directly tests the property we fixed and stays valid across platforms/BLAS
(a fixed-value golden of a GP-selection run could legitimately differ between
machines). See .claude/docs/model-selection-loop.md (hypothesis H5).
"""

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Ackley

from pysamoo.algorithms.csea import CSEA
from pysamoo.algorithms.ehvi import EHVI
from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.algorithms.krvea import KRVEA
from pysamoo.algorithms.moead_ego import MOEADEGO
from pysamoo.algorithms.parego import ParEGO
from pysamoo.algorithms.psaf import PSAF
from pysamoo.algorithms.saasbo import SAASBO
from pysamoo.algorithms.ssansga2 import SSANSGA2
from pysamoo.algorithms.tsemo import TSEMO
from pysamoo.algorithms.turbo import TuRBO


def _run(build):
    return minimize(build()[0], build()[1], build()[2], seed=1, verbose=False)


def _gpsaf():
    return (
        Ackley(n_var=5),
        GPSAF(GA(pop_size=10, n_offsprings=5), n_initial_doe=12, alpha=5, beta=10, n_max_infills=1),
        ("n_evals", 20),
    )


def _psaf():
    return (Ackley(n_var=5), PSAF(DE(pop_size=10, n_offsprings=5), alpha=5, beta=10), ("n_evals", 20))


def _ssansga2():
    return (
        ZDT1(n_var=5),
        SSANSGA2(n_initial_doe=12, n_infills=2, surr_pop_size=20, surr_n_gen=10),
        ("n_evals", 20),
    )


def _parego():
    return (ZDT1(n_var=5), ParEGO(n_initial_doe=12), ("n_evals", 20))


def _krvea():
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions

    ref = get_reference_directions("das-dennis", 3, n_partitions=4)
    algo = KRVEA(ref_dirs=ref, n_initial_doe=12, n_infills=2, w_max=5)
    return (get_problem("dtlz2", n_var=5, n_obj=3), algo, ("n_evals", 20))


def _ehvi():
    return (ZDT1(n_var=5), EHVI(n_initial_doe=12, pool=40, n_screen=6, n_samples=6), ("n_evals", 20))


def _turbo():
    return (Ackley(n_var=5), TuRBO(n_initial_doe=12, n_candidates=50), ("n_evals", 20))


def _moead_ego():
    return (ZDT1(n_var=5), MOEADEGO(n_initial_doe=12, n_infills=2, pool=40), ("n_evals", 20))


def _tsemo():
    return (ZDT1(n_var=5), TSEMO(n_initial_doe=12, n_infills=2, pool=40), ("n_evals", 20))


def _csea():
    from pymoo.problems import get_problem

    return (
        get_problem("dtlz2", n_var=5, n_obj=3),
        CSEA(n_initial_doe=12, n_infills=2, n_offspring=40),
        ("n_evals", 20),
    )


def _saasbo():
    return (Ackley(n_var=5), SAASBO(n_initial_doe=12), ("n_evals", 20))


@pytest.mark.parametrize(
    "build",
    [_gpsaf, _psaf, _ssansga2, _parego, _krvea, _ehvi, _turbo, _moead_ego, _tsemo, _csea, _saasbo],
    ids=["gpsaf", "psaf", "ssansga2", "parego", "krvea", "ehvi", "turbo", "moead_ego", "tsemo", "csea", "saasbo"],
)
def test_same_seed_is_reproducible(build):
    """Two runs with the same seed produce identical objective values."""
    r1 = _run(build)
    r2 = _run(build)
    f1 = np.atleast_2d(np.asarray(r1.F, dtype=float))
    f2 = np.atleast_2d(np.asarray(r2.F, dtype=float))
    assert f1.shape == f2.shape, "result shape differs across identical-seed runs"
    assert np.array_equal(f1, f2), "objective values differ across identical-seed runs"
