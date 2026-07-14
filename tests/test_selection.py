"""Tests for the pluggable model-selection strategies."""

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.single import Ackley

from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.core.selection import resolve
from pysamoo.core.target import Target


def test_resolve_names_and_factories():
    """resolve() maps names to factories and passes callables through."""
    assert resolve("full") is Target

    def custom(label, models):
        return Target(label, models)

    assert resolve(custom) is custom
    with pytest.raises(ValueError):
        resolve("does-not-exist")


def test_nth_validate_reduces_reselection():
    """Lazy re-selection (nth_validate>1) calls model selection fewer times."""
    import pysamoo.core.surrogate as surrogate_mod

    counter = {"n": 0}
    original = surrogate_mod.Surrogate.validate

    def spy(self, *args, **kwargs):
        counter["n"] += 1
        return original(self, *args, **kwargs)

    surrogate_mod.Surrogate.validate = spy
    try:

        def run(nth):
            counter["n"] = 0
            algo = GPSAF(GA(pop_size=10, n_offsprings=5), n_initial_doe=12, alpha=5, beta=10, nth_validate=nth)
            res = minimize(Ackley(n_var=5), algo, ("n_evals", 45), seed=1, verbose=False)
            return counter["n"], res

        n_every, res1 = run(1)
        n_lazy, res5 = run(5)
    finally:
        surrogate_mod.Surrogate.validate = original

    assert n_lazy < n_every, "nth_validate>1 should re-select fewer times"
    assert np.isfinite(res5.pop.get("F")).all()
