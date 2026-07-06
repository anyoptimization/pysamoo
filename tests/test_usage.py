"""Parametric smoke tests for every pysamoo usage scenario.

Each ``usage_*.py`` demo script carries a large presentation budget (hundreds of
evaluations) and ends in a ``matplotlib`` plot, so running the scripts verbatim
is slow and brittle against library drift. Instead we exercise the *same
algorithm on the same kind of problem* with the smallest meaningful budget
(initial DOE + ~1 infill) and no plotting. That validates every algorithm wires
up and produces a result in a few seconds total.
"""

import importlib.util

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.many import C3DTLZ4
from pymoo.problems.multi import SRN, ZDT1
from pymoo.problems.single import Ackley, Sphere
from pymoo.util.ref_dirs import get_reference_directions

from pysamoo.algorithms.ehvi import EHVI
from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.algorithms.krvea import KRVEA
from pysamoo.algorithms.parego import ParEGO
from pysamoo.algorithms.psaf import PSAF
from pysamoo.algorithms.ssansga2 import SSANSGA2
from pysamoo.algorithms.turbo import TuRBO

_HAS_SMAC = importlib.util.find_spec("smac") is not None


# --- one minimal builder per usage scenario; each returns the optimization result ---


def _gpsaf_single():
    """usage_gpsaf_single: GPSAF wrapping a GA on a single-objective problem."""
    algo = GPSAF(GA(pop_size=10, n_offsprings=5), n_initial_doe=10, alpha=5, beta=10, n_max_infills=1)
    return minimize(Ackley(n_var=5), algo, ("n_evals", 11), seed=1, verbose=False)


def _gpsaf_multi():
    """usage_gpsaf_multi: GPSAF wrapping NSGA2 on a bi-objective problem."""
    algo = GPSAF(NSGA2(pop_size=10, n_offsprings=5), n_initial_doe=10, alpha=5, beta=10, n_max_infills=1)
    return minimize(ZDT1(n_var=5), algo, ("n_evals", 11), seed=1, verbose=False)


def _gpsaf_many():
    """usage_gpsaf_many: GPSAF wrapping NSGA3 on a 3-objective problem."""
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=3)
    algo = GPSAF(NSGA3(ref_dirs, n_offsprings=5), n_initial_doe=12, alpha=5, beta=10, n_max_infills=1)
    return minimize(get_problem("dtlz2", n_var=6), algo, ("n_evals", 13), seed=1, verbose=False)


def _gpsaf_cmoo():
    """usage_gpsaf_cmoo: GPSAF wrapping NSGA3 on a constrained many-objective problem."""
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=3)
    algo = GPSAF(NSGA3(ref_dirs, n_offsprings=5), n_initial_doe=12, alpha=2, beta=10, n_max_infills=1)
    return minimize(C3DTLZ4(n_var=6), algo, ("n_evals", 13), seed=1, verbose=False)


def _gpsaf_constr():
    """usage_gpsaf_constr: GPSAF wrapping ISRES on a constrained single-objective problem."""
    from pymoo.algorithms.soo.nonconvex.isres import ISRES

    algo = GPSAF(ISRES(), n_initial_doe=10, alpha=3, beta=10, n_max_infills=1)
    return minimize(get_problem("g1"), algo, ("n_evals", 11), seed=1, verbose=False)


def _psaf():
    """usage_psaf: PSAF wrapping DE on a single-objective problem."""
    algo = PSAF(DE(pop_size=10, n_offsprings=5), alpha=5, beta=10)
    return minimize(Ackley(n_var=5), algo, ("n_evals", 11), seed=1, verbose=False)


def _ssansga2():
    """usage_ssansga2: steady-state surrogate-assisted NSGA-II."""
    algo = SSANSGA2(n_initial_doe=10, n_infills=1, surr_pop_size=20, surr_n_gen=10)
    return minimize(ZDT1(n_var=5), algo, ("n_evals", 11), seed=1, verbose=False)


def _parego():
    """usage_parego: ParEGO (scalarized multi-objective EGO) on a bi-objective problem."""
    return minimize(ZDT1(n_var=5), ParEGO(n_initial_doe=10), ("n_evals", 11), seed=1, verbose=False)


def _krvea():
    """usage_krvea: Kriging-assisted RVEA on a 3-objective problem."""
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions

    ref = get_reference_directions("das-dennis", 3, n_partitions=4)
    algo = KRVEA(ref_dirs=ref, n_initial_doe=10, n_infills=2, w_max=5)
    return minimize(get_problem("dtlz2", n_var=5, n_obj=3), algo, ("n_evals", 13), seed=1, verbose=False)


def _ehvi():
    """usage_ehvi: Expected Hypervolume Improvement BO on a bi-objective problem."""
    algo = EHVI(n_initial_doe=10, pool=40, n_screen=6, n_samples=6)
    return minimize(ZDT1(n_var=5), algo, ("n_evals", 12), seed=1, verbose=False)


def _turbo():
    """usage_turbo: trust-region Bayesian optimization on a single-objective problem."""
    return minimize(Ackley(n_var=5), TuRBO(n_initial_doe=10, n_candidates=50), ("n_evals", 12), seed=1, verbose=False)


def _lqcmaes():
    """usage_lqcmaes: surrogate-assisted (local quadratic) CMA-ES."""
    from pysamoo.vendor.lqcmaes import lqCMAES

    return minimize(Sphere(n_var=5), lqCMAES(), ("n_evals", 20), seed=1, verbose=False)


def _bo():
    """usage_bayesian_optimization: GP-based Bayesian optimization over one DACE surrogate."""
    from pysamoo.experimental.bo import BayesianOptimization

    return minimize(Sphere(n_var=5), BayesianOptimization(), ("n_gen", 2), seed=1, verbose=False)


def _smac():
    """usage_smac: SMAC wrapper (optional dependency)."""
    from pysamoo.vendor.smac import SMAC

    return minimize(Sphere(n_var=5), SMAC(), ("n_evals", 20), seed=1, verbose=False)


SCENARIOS = [
    pytest.param(_gpsaf_single, id="gpsaf_single"),
    pytest.param(_gpsaf_multi, id="gpsaf_multi"),
    pytest.param(_gpsaf_many, id="gpsaf_many"),
    pytest.param(_gpsaf_cmoo, id="gpsaf_cmoo"),
    pytest.param(_gpsaf_constr, id="gpsaf_constr"),
    pytest.param(_psaf, id="psaf"),
    pytest.param(_ssansga2, id="ssansga2"),
    pytest.param(_parego, id="parego"),
    pytest.param(_krvea, id="krvea"),
    pytest.param(_ehvi, id="ehvi"),
    pytest.param(_turbo, id="turbo"),
    pytest.param(_lqcmaes, id="lqcmaes"),
    pytest.param(_bo, id="bayesian_optimization"),
    pytest.param(
        _smac,
        id="smac",
        marks=pytest.mark.skipif(not _HAS_SMAC, reason="optional dependency 'smac' not installed"),
    ),
]


@pytest.mark.parametrize("build", SCENARIOS)
def test_usage_scenario(build):
    """The algorithm runs end-to-end on a minimal budget and evaluates solutions.

    We assert on the final population rather than ``res.F``: on a hard-constrained
    problem a tiny budget may not yet contain a feasible point (``res.F`` is then
    ``None``), but the run must still have evaluated finite objective values.
    """
    res = build()
    assert res is not None
    assert res.pop is not None and len(res.pop) > 0
    F = res.pop.get("F")
    assert F is not None and np.isfinite(F).all()


def test_constrained_sampling():
    """usage_constr_sampling: energy-based constrained sampling produces points in-bounds."""
    from pysamoo.sampling.energy import EnergyConstrainedSampling

    problem = SRN()

    def func_constr(X):
        G = problem.evaluate(X, return_values_of=["G"])
        return np.maximum(G, 0.0).sum(axis=1)

    X = EnergyConstrainedSampling(func_constr).do(problem, 20).get("X")
    xl, xu = problem.bounds()
    assert X.shape == (20, problem.n_var)
    assert (X >= xl).all() and (X <= xu).all()
