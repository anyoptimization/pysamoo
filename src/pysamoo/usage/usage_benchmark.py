"""Compare surrogate-assisted algorithms (and surrogate models) on a small budget.

Run with ``pyclawd python src/pysamoo/usage/usage_benchmark.py``. This is the
intended entry point for developing better methods: add a ``Scenario`` and read
the comparison table.
"""

from ezmodel.models.rbf import RBF
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi import ZDT1

from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.benchmark import (
    ProblemSpec,
    Scenario,
    format_table,
    make_surrogate,
    run_benchmark,
    summarize,
)

if __name__ == "__main__":
    problem = ZDT1(n_var=10)

    def base():
        return NSGA2(pop_size=20, n_offsprings=10)

    def only_rbf(**defaults):
        return {"rbf-cubic": RBF(kernel="cubic", **defaults)}

    scenarios = [
        Scenario("NSGA2 (baseline)", base),
        Scenario("GPSAF (Kriging)", lambda: GPSAF(base(), n_initial_doe=30)),
        Scenario(
            "GPSAF (RBF only)",
            lambda: GPSAF(base(), n_initial_doe=30, surrogate=make_surrogate(problem, obj_models=only_rbf)),
        ),
    ]

    problems = [ProblemSpec("ZDT1", problem, n_evals=200)]

    records = run_benchmark(scenarios, problems, n_seeds=3)
    print("\n" + format_table(summarize(records)))
