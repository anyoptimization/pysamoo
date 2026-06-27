"""Tests for the benchmarking harness."""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Sphere

from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.benchmark import (
    ProblemSpec,
    Scenario,
    format_table,
    make_surrogate,
    run_benchmark,
    score_run,
    summarize,
)


def test_run_benchmark_single_objective():
    """A single-objective benchmark produces finite f_gap scores and a summary."""
    scenarios = [
        Scenario("GA", lambda: GA(pop_size=10, n_offsprings=5)),
        Scenario("GPSAF", lambda: GPSAF(GA(pop_size=10, n_offsprings=5), n_initial_doe=10, n_max_infills=1)),
    ]
    problems = [ProblemSpec("Sphere", Sphere(n_var=5), n_evals=15)]

    records = run_benchmark(scenarios, problems, n_seeds=1, verbose=False)
    assert len(records) == 2
    assert all(r.metric == "f_gap" for r in records)
    assert all(np.isfinite(r.score) for r in records)

    summaries = summarize(records)
    assert len(summaries) == 2
    table = format_table(summaries)
    assert "Sphere" in table and "GPSAF" in table


def test_run_benchmark_multi_objective_igd():
    """A multi-objective benchmark scores with IGD against the Pareto front."""
    scenarios = [Scenario("NSGA2", lambda: NSGA2(pop_size=10, n_offsprings=5))]
    problems = [ProblemSpec("ZDT1", ZDT1(n_var=5), n_evals=20)]

    records = run_benchmark(scenarios, problems, n_seeds=1, verbose=False)
    assert records[0].metric == "igd"
    assert np.isfinite(records[0].score)


def test_make_surrogate_is_pluggable():
    """A custom model-set factory yields a usable surrogate for an algorithm."""
    from ezmodel.models.rbf import RBF

    problem = ZDT1(n_var=5)

    def only_rbf(**defaults):
        return {"rbf-cubic": RBF(kernel="cubic", **defaults)}

    surrogate = make_surrogate(problem, obj_models=only_rbf)
    assert len(surrogate.targets) == problem.n_obj

    algo = GPSAF(NSGA2(pop_size=10, n_offsprings=5), n_initial_doe=10, n_max_infills=1, surrogate=surrogate)
    records = run_benchmark(
        [Scenario("GPSAF+RBF", lambda: algo)], [ProblemSpec("ZDT1", problem, 11)], n_seeds=1, verbose=False
    )
    assert np.isfinite(records[0].score)


def test_score_run_handles_infeasible():
    """An empty/None result is scored as infinite and infeasible."""

    class _FakeRes:
        F = None
        algorithm = None

    metric, value, feasible = score_run(Sphere(n_var=3), _FakeRes())
    assert value == float("inf")
    assert feasible is False
