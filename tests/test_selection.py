"""Tests for the pluggable model-selection strategies."""

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.single import Ackley

from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.core.racing import RacingTarget, family_of
from pysamoo.core.selection import resolve
from pysamoo.core.target import Target


def test_resolve_names_and_factories():
    """resolve() maps names to factories and passes callables through."""
    assert resolve("full") is Target
    assert resolve("racing") is RacingTarget

    def custom(label, models):
        return Target(label, models)

    assert resolve(custom) is custom
    with pytest.raises(ValueError):
        resolve("does-not-exist")


def test_family_of():
    """family_of groups RBF by kernel and all Kriging together."""
    assert family_of("RBF[kernel=gaussian,tail=linear,normalized=True]") == "gaussian"
    assert family_of("kriging-quadr-ARD") == "kriging"


def test_racing_selection_runs_and_is_valid():
    """GPSAF with the racing strategy optimizes end-to-end and returns finite F."""
    algo = GPSAF(
        GA(pop_size=10, n_offsprings=5), n_initial_doe=12, alpha=5, beta=10, n_max_infills=1, selection="racing"
    )
    res = minimize(Ackley(n_var=5), algo, ("n_evals", 22), seed=1, verbose=False)
    assert res.pop is not None and len(res.pop) > 0
    F = res.pop.get("F")
    assert np.isfinite(F).all()


def test_racing_pool_shrinks():
    """The active set drops below the initial pool size during a run."""
    sizes = []
    original = RacingTarget.validate

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        sizes.append(len(self.active))
        return out

    RacingTarget.validate = spy
    try:
        # nth_validate=1 so the racing target is exercised every iteration (isolating
        # the shrink behaviour from the lazy re-selection gate).
        algo = GPSAF(
            GA(pop_size=10, n_offsprings=5), n_initial_doe=12, alpha=5, beta=10, nth_validate=1, selection="racing"
        )
        minimize(Ackley(n_var=5), algo, ("n_evals", 30), seed=1, verbose=False)
    finally:
        RacingTarget.validate = original

    assert sizes, "racing validate was never called"
    assert min(sizes) < sizes[0], "active pool never shrank"


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
